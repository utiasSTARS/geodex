/// @file scripts/robotgen/pinocchio_codegen.cpp
/// @brief Generates the CRBA source of one robot with Pinocchio and CppAD::CG.
///
///   pinocchio_codegen <urdf_path> <robot_name> <out_dir>
///
/// Traces `pinocchio::crba` for a URDF and writes straight-line C source to
/// <out_dir>/<robot_name>_crba.cpp and .hpp, with the entry point
///
///   void <robot>_crba(const double q[nq], double M_upper[nv*(nv+1)/2]);
///
/// `M_upper` is the row-major upper triangle of the mass matrix. Both files record the
/// SHA-256 of the URDF on a `urdf-sha256:` line.
///
/// A URDF mimic joint follows its primary joint, `q_j = multiplier * q_primary + offset`.
/// With mimic joints, the function takes the independent coordinates `r` and returns
/// `A^T M(A r + b) A`, where `A` and `b` collect the mimic relations.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <pinocchio/codegen/code-generator-algo.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include <cppad/cg/model/save_files_model_library_processor.hpp>

namespace fs = std::filesystem;

// Expose the protected library source generator of CodeGenBase, which
// SaveFilesModelLibraryProcessor needs.
template <typename Scalar>
struct CodeGenCRBAExposed : public ::pinocchio::CodeGenCRBA<Scalar> {
  using Parent = ::pinocchio::CodeGenCRBA<Scalar>;
  using Base = ::pinocchio::CodeGenBase<Scalar>;
  using Parent::Parent;
  CppAD::cg::ModelLibraryCSourceGen<Scalar>& libgen() { return *Base::libcgen_ptr; }
};

// CRBA in independent coordinates r of a model with coupled joints q = A r + b.
template <typename Scalar>
struct CodeGenCoupledCRBA : public ::pinocchio::CodeGenBase<Scalar> {
  using Base = ::pinocchio::CodeGenBase<Scalar>;
  using typename Base::ADConfigVectorType;
  using typename Base::ADMatrixXs;
  using typename Base::ADScalar;
  using typename Base::MatrixXs;
  using typename Base::Model;
  using typename Base::VectorXs;

  CodeGenCoupledCRBA(const Model& model, const MatrixXs& A, const VectorXs& b,
                     const std::string& function_name, const std::string& library_name)
      : Base(model, A.cols(), (A.cols() * (A.cols() + 1)) / 2, function_name, library_name),
        A_(A),
        b_(b) {
    Base::build_jacobian = false;
  }

  void buildMap() override {
    CppAD::Independent(Base::ad_X);
    const ADMatrixXs A = A_.template cast<ADScalar>();
    const ADConfigVectorType q = A * Base::ad_X + b_.template cast<ADScalar>();
    ::pinocchio::crba(Base::ad_model, Base::ad_data, q, ::pinocchio::Convention::WORLD);
    const ADMatrixXs M = Base::ad_data.M.template selfadjointView<Eigen::Upper>();
    const ADMatrixXs Mc = A.transpose() * M * A;
    Eigen::DenseIndex k = 0;
    for (Eigen::DenseIndex i = 0; i < Mc.rows(); ++i) {
      for (Eigen::DenseIndex j = i; j < Mc.cols(); ++j) Base::ad_Y[k++] = Mc(i, j);
    }
    Base::ad_fun.Dependent(Base::ad_X, Base::ad_Y);
    Base::ad_fun.optimize("no_compare_op");
  }

  CppAD::cg::ModelLibraryCSourceGen<Scalar>& libgen() { return *Base::libcgen_ptr; }

 private:
  MatrixXs A_;
  VectorXs b_;
};

namespace {

/// Independent coordinates of a URDF with mimic joints, `q = A r + b`.
struct Coupling {
  Eigen::MatrixXd A;
  Eigen::VectorXd b;
  Eigen::VectorXd lower;
  Eigen::VectorXd upper;
  std::vector<std::string> description;  // one line per mimic joint
};

/// Read `<mimic>` tags as joint -> (primary, multiplier, offset).
auto read_mimic_joints(const fs::path& urdf)
    -> std::map<std::string, std::tuple<std::string, double, double>> {
  std::ifstream in(urdf);
  std::stringstream ss;
  ss << in.rdbuf();
  const std::string xml = ss.str();
  const std::regex joint_re(R"re(<joint\b([^>]*)>([\s\S]*?)</joint>)re");
  const std::regex mimic_re(R"re(<mimic\b([^>]*)/?>)re");
  const auto attr = [](const std::string& attrs, const std::string& key) -> std::string {
    const std::regex re("\\b" + key + R"re(\s*=\s*"([^"]*)")re");
    std::smatch m;
    return std::regex_search(attrs, m, re) ? m[1].str() : std::string{};
  };
  std::map<std::string, std::tuple<std::string, double, double>> mimics;
  for (auto it = std::sregex_iterator(xml.begin(), xml.end(), joint_re);
       it != std::sregex_iterator(); ++it) {
    const std::string body = (*it)[2].str();
    std::smatch m;
    if (!std::regex_search(body, m, mimic_re)) continue;
    const std::string mult = attr(m[1].str(), "multiplier");
    const std::string off = attr(m[1].str(), "offset");
    mimics[attr((*it)[1].str(), "name")] = {attr(m[1].str(), "joint"),
                                             mult.empty() ? 1.0 : std::stod(mult),
                                             off.empty() ? 0.0 : std::stod(off)};
  }
  return mimics;
}

/// Build the coupling of @p model from the URDF mimic tags, or an empty A without them.
auto make_coupling(const ::pinocchio::Model& model, const fs::path& urdf) -> Coupling {
  // Keep the tags of the model's own joints. A fixed joint can keep a mimic tag.
  std::map<std::string, std::tuple<std::string, double, double>> mimics;
  for (const auto& [name, mimic] : read_mimic_joints(urdf)) {
    if (model.existJointName(name)) mimics.emplace(name, mimic);
  }
  Coupling c;
  if (mimics.empty()) return c;
  std::vector<std::string> independent;
  for (::pinocchio::JointIndex j = 1; j < model.joints.size(); ++j) {
    if (model.joints[j].nq() != 1 || model.joints[j].nv() != 1) {
      throw std::runtime_error("coupled CRBA supports one-coordinate joints only: " +
                               model.names[j]);
    }
    if (!mimics.count(model.names[j])) independent.push_back(model.names[j]);
  }
  const auto col = [&](const std::string& name) {
    const auto it = std::find(independent.begin(), independent.end(), name);
    if (it == independent.end()) throw std::runtime_error("mimic primary not independent: " + name);
    return static_cast<int>(it - independent.begin());
  };
  c.A = Eigen::MatrixXd::Zero(model.nq, static_cast<Eigen::Index>(independent.size()));
  c.b = Eigen::VectorXd::Zero(model.nq);
  c.lower.resize(static_cast<Eigen::Index>(independent.size()));
  c.upper.resize(static_cast<Eigen::Index>(independent.size()));
  for (::pinocchio::JointIndex j = 1; j < model.joints.size(); ++j) {
    const int iq = model.joints[j].idx_q();
    const auto it = mimics.find(model.names[j]);
    if (it == mimics.end()) {
      const int r = col(model.names[j]);
      c.A(iq, r) = 1.0;
      c.lower[r] = model.lowerPositionLimit[iq];
      c.upper[r] = model.upperPositionLimit[iq];
    } else {
      const auto& [primary, multiplier, offset] = it->second;
      c.A(iq, col(primary)) = multiplier;
      c.b[iq] = offset;
      std::ostringstream line;
      line.precision(17);
      line << model.names[j] << " = " << multiplier << " * " << primary << " + " << offset;
      c.description.push_back(line.str());
    }
  }
  return c;
}

// SHA-256 (FIPS 180-4) for the `urdf-sha256:` line.
class Sha256 {
 public:
  Sha256() { reset(); }

  void update(const std::uint8_t* data, std::size_t len) {
    for (std::size_t i = 0; i < len; ++i) {
      buf_[bufLen_++] = data[i];
      if (bufLen_ == 64) {
        transform(buf_);
        bitLen_ += 512;
        bufLen_ = 0;
      }
    }
  }

  std::string hex() {
    // Take the message length before the padding completes a block.
    const std::uint64_t messageBits = bitLen_ + static_cast<std::uint64_t>(bufLen_) * 8u;
    std::uint8_t pad[64] = {0};
    pad[0] = 0x80;
    if (bufLen_ < 56) {
      update(pad, 56 - bufLen_);
    } else {
      update(pad, 64 - bufLen_);
      update(reinterpret_cast<std::uint8_t*>(std::memset(pad, 0, 64)), 56);
    }
    std::uint8_t lenBytes[8];
    for (int i = 0; i < 8; ++i) lenBytes[7 - i] = (messageBits >> (i * 8)) & 0xff;
    update(lenBytes, 8);
    std::ostringstream out;
    out << std::hex;
    for (int i = 0; i < 8; ++i) {
      out.width(8);
      out.fill('0');
      out << state_[i];
    }
    return out.str();
  }

 private:
  static constexpr std::uint32_t K_[64] = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
      0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
      0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
      0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
      0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
      0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
      0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
      0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
      0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
      0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
      0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

  void reset() {
    state_[0] = 0x6a09e667;
    state_[1] = 0xbb67ae85;
    state_[2] = 0x3c6ef372;
    state_[3] = 0xa54ff53a;
    state_[4] = 0x510e527f;
    state_[5] = 0x9b05688c;
    state_[6] = 0x1f83d9ab;
    state_[7] = 0x5be0cd19;
    bitLen_ = 0;
    bufLen_ = 0;
  }

  static std::uint32_t rotr(std::uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
  }

  void transform(const std::uint8_t* chunk) {
    std::uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      w[i] = (std::uint32_t(chunk[4 * i]) << 24) |
             (std::uint32_t(chunk[4 * i + 1]) << 16) |
             (std::uint32_t(chunk[4 * i + 2]) << 8) |
             std::uint32_t(chunk[4 * i + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      const auto s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const auto s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    auto a = state_[0], b = state_[1], c = state_[2], d = state_[3];
    auto e = state_[4], f = state_[5], g = state_[6], h = state_[7];
    for (int i = 0; i < 64; ++i) {
      const auto S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const auto ch = (e & f) ^ (~e & g);
      const auto t1 = h + S1 + ch + K_[i] + w[i];
      const auto S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const auto mj = (a & b) ^ (a & c) ^ (b & c);
      const auto t2 = S0 + mj;
      h = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::uint32_t state_[8];
  std::uint64_t bitLen_;
  std::uint8_t buf_[64];
  std::size_t bufLen_;
};

constexpr std::uint32_t Sha256::K_[64];

std::string sha256_of_file(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + path.string());
  Sha256 h;
  std::vector<std::uint8_t> buf(8192);
  while (in.read(reinterpret_cast<char*>(buf.data()), buf.size()) || in.gcount() > 0) {
    h.update(buf.data(), static_cast<std::size_t>(in.gcount()));
  }
  return h.hex();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " <urdf_path> <robot_name> <out_dir>\n";
    return 1;
  }
  const fs::path urdf_path = argv[1];
  const std::string robot_name = argv[2];
  const fs::path out_dir = argv[3];

  if (!fs::exists(urdf_path)) {
    std::cerr << "URDF not found: " << urdf_path << "\n";
    return 1;
  }
  fs::create_directories(out_dir);

  ::pinocchio::Model model;
  ::pinocchio::urdf::buildModel(urdf_path.string(), model);
  std::cout << "Loaded model '" << model.name << "' with nq=" << model.nq
            << ", nv=" << model.nv << "\n";

  const std::string fn_name = robot_name + "_crba";
  const std::string fwd_name = fn_name + "_forward_zero";
  const Coupling coupling = make_coupling(model, urdf_path);
  const bool coupled = coupling.A.size() > 0;
  const int nq = coupled ? static_cast<int>(coupling.A.cols()) : model.nq;
  const int nv = coupled ? nq : model.nv;
  const Eigen::VectorXd lower = coupled ? coupling.lower : Eigen::VectorXd(model.lowerPositionLimit);
  const Eigen::VectorXd upper = coupled ? coupling.upper : Eigen::VectorXd(model.upperPositionLimit);
  if (coupled) {
    std::cout << "Coupled coordinates: " << nq << " independent of " << model.nq << "\n";
    for (const auto& line : coupling.description) std::cout << "  " << line << "\n";
  }

  std::unique_ptr<CodeGenCRBAExposed<double>> plain;
  std::unique_ptr<CodeGenCoupledCRBA<double>> coupled_gen;
  CppAD::cg::ModelLibraryCSourceGen<double>* libgen = nullptr;
  if (coupled) {
    coupled_gen = std::make_unique<CodeGenCoupledCRBA<double>>(model, coupling.A, coupling.b,
                                                               fn_name, fn_name + "_lib");
    coupled_gen->initLib();
    libgen = &coupled_gen->libgen();
  } else {
    plain = std::make_unique<CodeGenCRBAExposed<double>>(model, fn_name, fn_name + "_lib");
    plain->initLib();
    libgen = &plain->libgen();
  }

  // Write the CppAD::CG sources into a scratch directory, then concatenate them. Without the
  // Jacobian, CppAD::CG writes forward_zero and a few support files.
  const fs::path scratch = out_dir / "_cg_scratch";
  fs::remove_all(scratch);
  fs::create_directories(scratch);
  CppAD::cg::SaveFilesModelLibraryProcessor<double>::saveLibrarySourcesTo(
      *libgen, scratch.string());

  std::vector<fs::path> source_files;
  for (const auto& entry : fs::recursive_directory_iterator(scratch)) {
    if (!entry.is_regular_file()) continue;
    const auto ext = entry.path().extension();
    if (ext == ".c" || ext == ".h" || ext == ".hpp") {
      source_files.push_back(entry.path());
    }
  }
  std::sort(source_files.begin(), source_files.end());
  std::cout << "CppAD::CG produced " << source_files.size() << " source file(s):\n";
  for (const auto& p : source_files) {
    std::cout << "  " << fs::relative(p, scratch).string()
              << "  (" << fs::file_size(p) << " bytes)\n";
  }

  const std::string urdf_hash = sha256_of_file(urdf_path);
  const fs::path out_cpp = out_dir / (robot_name + "_crba.cpp");
  const fs::path out_hpp = out_dir / (robot_name + "_crba.hpp");
  const int upper_count = (nv * (nv + 1)) / 2;

  // The .cpp holds every CppAD::CG source verbatim and an extern "C" wrapper around
  // forward_zero with the simple signature.
  std::ofstream cpp(out_cpp);
  cpp << "// AUTO-GENERATED by scripts/robotgen/pinocchio_codegen. DO NOT EDIT.\n"
      << "// Source URDF: " << urdf_path.filename().string() << "\n"
      << "// urdf-sha256: " << urdf_hash << "\n"
      << "// nq = " << nq << ", nv = " << nv
      << ", upper-triangle entries = " << upper_count << "\n"
      << "// Regenerate with scripts/robotgen/generate.sh <robot>.\n";
  if (coupled) {
    cpp << "// Independent coordinates; the mimic joints follow them:\n";
    for (const auto& line : coupling.description) cpp << "//   " << line << "\n";
  }
  cpp << "//\n"
      << "// Generated by Pinocchio CodeGenCRBA (CppAD::CG). Output layout\n"
      << "// is row-major upper triangle of the joint-space mass matrix:\n"
      << "//   M_upper[0]            = M(0,0)\n"
      << "//   M_upper[1..nv-1]      = M(0,1), M(0,2), ..., M(0,nv-1)\n"
      << "//   M_upper[nv]           = M(1,1)\n"
      << "//   ...\n"
      << "//   M_upper[end]          = M(nv-1, nv-1)\n"
      << "\n"
      << "#include <math.h>\n"
      << "#include <stdint.h>\n"
      << "#include <stdlib.h>\n"
      << "\n";

  cpp << "// ---- generated source files (verbatim, wrapped in extern \"C\" for C linkage) ----\n";
  cpp << "extern \"C\" {\n\n";
  for (const auto& p : source_files) {
    std::ifstream in(p);
    cpp << "// === " << p.filename().string() << " ===\n" << in.rdbuf() << "\n";
  }
  cpp << "}  // extern \"C\"\n\n";
  cpp << "// ---- thin C wrapper to a stable, simple signature ----\n";
  cpp << "extern \"C\" void " << fn_name << "(const double q[" << nq
      << "], double M_upper[" << upper_count << "]) {\n"
      << "  const double* in[1]  = { q };\n"
      << "  double*       out[1] = { M_upper };\n"
      << "  // No atomic functions inside CRBA; zero-init the dispatch table.\n"
      << "  struct LangCAtomicFun atomic_fun = {};\n"
      << "  " << fwd_name << "(in, out, atomic_fun);\n"
      << "}\n";
  cpp.close();

  // The header holds the declaration, nq, nv and the joint limits from the URDF.
  std::ofstream hpp(out_hpp);
  hpp << "// AUTO-GENERATED by scripts/robotgen/pinocchio_codegen. DO NOT EDIT.\n"
      << "// Source URDF: " << urdf_path.filename().string() << "\n"
      << "// urdf-sha256: " << urdf_hash << "\n\n"
      << "#pragma once\n\n"
      << "extern \"C\" void " << fn_name << "(const double q[" << nq
      << "], double M_upper[" << upper_count << "]);\n\n"
      << "namespace geodex::robots::generated {\n"
      << "constexpr int " << robot_name << "_nq = " << nq << ";\n"
      << "constexpr int " << robot_name << "_nv = " << nv << ";\n"
      << "constexpr int " << robot_name << "_upper_count = " << upper_count << ";\n";

  hpp.precision(17);
  hpp << "constexpr double " << robot_name << "_lower_limit[" << nq << "] = {";
  for (int i = 0; i < nq; ++i) {
    hpp << (i == 0 ? "" : ", ") << lower(i);
  }
  hpp << "};\n";

  hpp << "constexpr double " << robot_name << "_upper_limit[" << nq << "] = {";
  for (int i = 0; i < nq; ++i) {
    hpp << (i == 0 ? "" : ", ") << upper(i);
  }
  hpp << "};\n";

  hpp << "}  // namespace geodex::robots::generated\n";
  hpp.close();

  fs::remove_all(scratch);
  std::cout << "Wrote " << out_cpp << " and " << out_hpp << "\n";
  std::cout << "URDF SHA-256: " << urdf_hash << "\n";
  return 0;
}
