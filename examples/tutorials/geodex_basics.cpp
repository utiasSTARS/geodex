// Examples for the geodex basics tutorial, the C++ version of geodex_basics.py.
//
// Each step of the tutorial runs in a scope of its own.
//
// Usage: geodex_basics [--json out.json]

#include <cmath>

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// [docs-start:setup]
#include <iostream>

#include <geodex/geodex.hpp>
// [docs-end:setup]

#include "../common/json_output.hpp"

using geodex_examples::Json;

int main(int argc, char** argv) {
  std::vector<std::pair<std::string, Json>> out;

  {
    // [docs-start:first-manifold]
    geodex::Sphere<> sphere;
    std::cout << "dim = " << sphere.dim() << "\n";  // 2
    // [docs-end:first-manifold]
    out.emplace_back("first_manifold", Json(sphere.dim()));
  }

  {
    // [docs-start:fully-qualified]
    geodex::Sphere<2, geodex::SphereRoundMetric, geodex::SphereExponentialMap,
                   geodex::ScrambledHaltonSampler>
        sphere;
    // [docs-end:fully-qualified]
    static_assert(std::is_same_v<decltype(sphere), geodex::Sphere<>>);
  }

  {
    // [docs-start:create-manifolds]
    geodex::Euclidean<3> euclidean;  // R^3 with the standard metric
    geodex::Torus<2> torus;          // 2-torus with the flat metric
    // [docs-end:create-manifolds]
    out.emplace_back("create_manifolds",
                     Json::array({euclidean.dim(), torus.dim()}));
  }

  {
    // [docs-start:inner-sphere]
    geodex::Sphere<> sphere;

    Eigen::Vector3d p{0.0, 0.0, 1.0};  // north pole
    Eigen::Vector3d u{1.0, 0.0, 0.0};  // tangent vector pointing east
    Eigen::Vector3d v{0.0, 1.0, 0.0};  // tangent vector pointing south

    double ip = sphere.inner(p, u, v);
    std::cout << "ip = " << ip << "\n";  // 0.0 (orthogonal)
    double n = sphere.norm(p, u);
    std::cout << "n = " << n << "\n";  // 1.0
    // [docs-end:inner-sphere]
    out.emplace_back("inner_sphere", Json::array({ip, n}));
  }

  {
    // [docs-start:inner-euclidean]
    geodex::Euclidean<3> euclidean;

    Eigen::Vector3d p{0.0, 0.0, 0.0};
    Eigen::Vector3d u{1.0, 0.0, 0.0};
    Eigen::Vector3d v{0.0, 1.0, 0.0};

    double ip = euclidean.inner(p, u, v);
    std::cout << "ip = " << ip << "\n";  // 0.0
    double n = euclidean.norm(p, u);
    std::cout << "n = " << n << "\n";  // 1.0
    // [docs-end:inner-euclidean]
    out.emplace_back("inner_euclidean", Json::array({ip, n}));
  }

  {
    // [docs-start:exp-log-sphere]
    geodex::Sphere<> sphere;

    Eigen::Vector3d p{0.0, 0.0, 1.0};  // north pole
    Eigen::Vector3d q{1.0, 0.0, 0.0};  // point on the equator
    Eigen::IOFormat fmt(4, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]");

    // log gives the tangent vector at p that points toward q.
    Eigen::Vector3d v = sphere.log(p, q);
    std::cout << "v = " << v.transpose().format(fmt) << "\n";  // about [pi/2, 0, 0]

    // exp follows that tangent vector and recovers q.
    Eigen::Vector3d q_recovered = sphere.exp(p, v);
    std::cout << "q_recovered = " << q_recovered.transpose().format(fmt)
              << "\n";  // about [1, 0, 0]
    // [docs-end:exp-log-sphere]
    out.emplace_back("exp_log_sphere", Json::array({Json(v), Json(q_recovered)}));
  }

  {
    // [docs-start:exp-log-euclidean]
    geodex::Euclidean<3> euclidean;

    Eigen::Vector3d p{1.0, 0.0, 0.0};
    Eigen::Vector3d q{0.0, 1.0, 0.0};
    Eigen::IOFormat fmt(4, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]");

    Eigen::Vector3d v = euclidean.log(p, q);
    std::cout << "v = " << v.transpose().format(fmt) << "\n";  // [-1, 1, 0]
    Eigen::Vector3d q2 = euclidean.exp(p, v);
    std::cout << "q2 = " << q2.transpose().format(fmt)
              << "\n";  // [0, 1, 0], which is q
    // [docs-end:exp-log-euclidean]
    out.emplace_back("exp_log_euclidean", Json::array({Json(v), Json(q2)}));
  }

  {
    // [docs-start:distance-sphere]
    geodex::Sphere<> sphere;

    Eigen::Vector3d p{0.0, 0.0, 1.0};  // north pole
    Eigen::Vector3d q{1.0, 0.0, 0.0};  // equator

    double d = sphere.distance(p, q);
    std::cout << "d = " << d << "\n";  // 1.5708, about pi/2
    // [docs-end:distance-sphere]
    out.emplace_back("distance_sphere", Json(d));
  }

  {
    // [docs-start:distance-euclidean]
    geodex::Euclidean<3> euclidean;

    Eigen::Vector3d p{1.0, 0.0, 0.0};
    Eigen::Vector3d q{0.0, 1.0, 0.0};

    double d = euclidean.distance(p, q);
    std::cout << "d = " << d << "\n";  // 1.41421, about sqrt(2)
    // [docs-end:distance-euclidean]
    out.emplace_back("distance_euclidean", Json(d));
  }

  {
    // [docs-start:distance-torus]
    geodex::Torus<1> circle;

    Eigen::Vector<double, 1> p{0.1};
    Eigen::Vector<double, 1> q{6.0};

    double d = circle.distance(p, q);
    std::cout << "d = " << d << "\n";  // 0.383, about 2 pi - 5.9
    // [docs-end:distance-torus]
    out.emplace_back("distance_torus", Json(d));
  }

  {
    // [docs-start:geodesic-sphere]
    geodex::Sphere<> sphere;

    Eigen::Vector3d p{0.0, 0.0, 1.0};  // north pole
    Eigen::Vector3d q{1.0, 0.0, 0.0};  // equator
    Eigen::IOFormat fmt(4, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]");

    Eigen::Vector3d mid = sphere.geodesic(p, q, 0.5);
    std::cout << "mid = " << mid.transpose().format(fmt)
              << "\n";  // about [0.707, 0, 0.707]

    // Trace the whole geodesic
    Eigen::Vector3d pt;
    for (int i = 0; i <= 10; ++i) {
      double t = i / 10.0;
      pt = sphere.geodesic(p, q, t);
      std::cout << "t=" << t << ": " << pt.transpose().format(fmt) << "\n";
    }
    // [docs-end:geodesic-sphere]
    out.emplace_back("geodesic_sphere", Json::array({Json(mid), Json(pt)}));
  }

  {
    // [docs-start:geodesic-euclidean]
    geodex::Euclidean<3> euclidean;

    Eigen::Vector3d p{0.0, 0.0, 0.0};
    Eigen::Vector3d q{2.0, 4.0, 6.0};
    Eigen::IOFormat fmt(4, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]");

    Eigen::Vector3d mid = euclidean.geodesic(p, q, 0.5);
    std::cout << "mid = " << mid.transpose().format(fmt) << "\n";  // [1, 2, 3]
    // [docs-end:geodesic-euclidean]
    out.emplace_back("geodesic_euclidean", Json(mid));
  }

  {
    // [docs-start:random-points]
    geodex::Sphere<> sphere;
    geodex::Euclidean<3> euclidean;
    geodex::Torus<2> torus;

    auto p1 = sphere.random_point();     // uniform on the sphere
    auto p2 = euclidean.random_point();  // uniform in [-1, 1]^3
    auto p3 = torus.random_point();      // uniform in [0, 2 pi)^2
    // [docs-end:random-points]
    out.emplace_back("random_points", Json::array({static_cast<int>(p1.size()),
                                                   static_cast<int>(p2.size()),
                                                   static_cast<int>(p3.size())}));

    // [docs-start:sampler-choice]
    // Euclidean space with the pseudo-random sampler instead of the default
    // scrambled Halton sequence.
    geodex::Euclidean<3, geodex::EuclideanStandardMetric<3>,
                      geodex::PseudoRandomSampler>
        r3;
    // [docs-end:sampler-choice]

    // [docs-start:seeding]
    geodex::set_default_seed(42);  // the shared source of default samplers
    sphere.seed(42);               // or one manifold's own sampler
    // [docs-end:seeding]
    out.emplace_back("seeding", Json(Eigen::Vector3d(sphere.random_point())));
  }

  {
    // [docs-start:torus-wrap]
    geodex::Torus<2> torus;

    Eigen::Vector2d p{0.1, 0.2};
    Eigen::Vector2d q{6.1, 0.5};
    Eigen::IOFormat fmt(4, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]");

    // log wraps to the shortest path
    Eigen::Vector2d v = torus.log(p, q);
    std::cout << "v = " << v.transpose().format(fmt)
              << "\n";  // about [-0.283, 0.3]

    // exp follows the tangent vector and wraps back to [0, 2 pi)
    Eigen::Vector2d q_recovered = torus.exp(p, v);
    std::cout << "q_recovered = " << q_recovered.transpose().format(fmt)
              << "\n";  // about [6.1, 0.5]
    // [docs-end:torus-wrap]
    out.emplace_back("torus_wrap", Json::array({Json(v), Json(q_recovered)}));
  }

  {
    // [docs-start:so3]
    geodex::SO3<> so3;

    Eigen::Vector4d q0 = so3.random_point();  // unit quaternion [x, y, z, w]
    Eigen::Vector4d q1 = so3.random_point();

    Eigen::Vector3d w = so3.log(q0, q1);              // body angular velocity
    Eigen::Vector4d mid = so3.geodesic(q0, q1, 0.5);  // SLERP midpoint
    // [docs-end:so3]
  }

  {
    // [docs-start:se2]
    geodex::SE2<> se2;

    Eigen::Vector3d p{1.0, 2.0, 0.0};   // pose at (1, 2), heading east
    Eigen::Vector3d q{3.0, 4.0, 1.57};  // pose at (3, 4), heading north
    Eigen::IOFormat fmt(4, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]");

    Eigen::Vector3d v = se2.log(p, q);
    std::cout << "log: " << v.transpose().format(fmt) << "\n";

    Eigen::Vector3d q_recovered = se2.exp(p, v);
    std::cout << "recovered: " << q_recovered.transpose().format(fmt) << "\n";

    double d = se2.distance(p, q);
    std::cout << "distance: " << d << "\n";
    // [docs-end:se2]
    out.emplace_back("se2", Json::array({Json(v), Json(q_recovered), Json(d)}));
  }

  {
    // [docs-start:se2-car]
    // A large w_y makes sideways motion expensive, as for a wheeled base.
    geodex::SE2LeftInvariantMetric car_metric{1.0, 100.0, 1.0};
    geodex::SE2<> se2_car{car_metric};

    Eigen::Vector3d p{0.0, 0.0, 0.0};  // facing east
    Eigen::Vector3d q{2.0, 2.0, 0.0};  // same heading, offset diagonally

    double d = se2_car.distance(p, q);
    std::cout << "distance: " << d << "\n";
    // [docs-end:se2-car]
    out.emplace_back("se2_car", Json(d));
  }

  {
    // [docs-start:se3]
    geodex::SE3<> se3;

    Eigen::Matrix<double, 7, 1> a = se3.random_point();  // pose [t; q]
    Eigen::Matrix<double, 7, 1> b = se3.random_point();

    Eigen::Matrix<double, 6, 1> xi = se3.log(a, b);  // twist [v; w]
    Eigen::Matrix<double, 7, 1> mid =
        se3.geodesic(a, b, 0.5);  // screw-motion midpoint
    // [docs-end:se3]
  }

  {
    // [docs-start:so3-frames]
    using BodySO3 =
        geodex::SO3<geodex::SO3CanonicalMetric, geodex::SO3LeftExponentialMap>;
    using WorldSO3 =
        geodex::SO3<geodex::SO3CanonicalMetric, geodex::SO3RightExponentialMap>;

    Eigen::Vector4d q0{0.5, 0.0, 0.0, 0.8660};     // 60 degrees about x
    Eigen::Vector4d q1{0.0, 0.0, 0.7071, 0.7071};  // 90 degrees about z

    // The metric is bi-invariant, and both frames report the same distance.
    double db = BodySO3{}.distance(q0, q1);   // 1.8235
    double dw = WorldSO3{}.distance(q0, q1);  // 1.8235
    // [docs-end:so3-frames]
    out.emplace_back("so3_frames", Json::array({db, dw}));
  }

  {
    // [docs-start:se3-frames]
    using BodySE3 =
        geodex::SE3<geodex::SE3InvariantMetric, geodex::SE3LeftExponentialMap>;
    using WorldSE3 =
        geodex::SE3<geodex::SE3InvariantMetric, geodex::SE3RightExponentialMap>;

    Eigen::Matrix<double, 7, 1> a, b;
    a << 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.8660;  // pose [t; q]
    b << 0.0, 1.0, 2.0, 0.0, 0.0, 0.7071, 0.7071;

    // SE(3) does not have a bi-invariant metric, and the frames report different
    // distances.
    double db = BodySE3{}.distance(a, b);   // 3.2424
    double dw = WorldSE3{}.distance(a, b);  // 2.8023
    // [docs-end:se3-frames]
    out.emplace_back("se3_frames", Json::array({db, dw}));
  }

  {
    // [docs-start:product]
    auto space = geodex::make_product(geodex::Euclidean<3>{}, geodex::SE2<>{});
    std::cout << "dim = " << space.dim() << "\n";  // 6

    auto c =
        space.random_point();  // one joint low-discrepancy sample over both blocks
    // [docs-end:product]
    out.emplace_back("product", Json(space.dim()));
  }

  {
    // [docs-start:constant-spd]
    Eigen::Matrix3d A = Eigen::Vector3d(4.0, 1.0, 1.0).asDiagonal();
    geodex::ConstantSPDMetric<3> weighted{A};

    geodex::Euclidean<3, geodex::ConstantSPDMetric<3>> r3_weighted{weighted};

    Eigen::Vector3d p{0.0, 0.0, 0.0};
    Eigen::Vector3d u{1.0, 0.0, 0.0};
    Eigen::Vector3d v{0.0, 1.0, 0.0};

    double ip = r3_weighted.inner(p, u, v);  // 0.0, still orthogonal
    double n = r3_weighted.norm(p, u);       // 2.0, scaled by sqrt(4)

    Eigen::Vector3d q{1.0, 1.0, 1.0};
    double d = r3_weighted.distance(p, q);
    std::cout << "d = " << d << "\n";  // sqrt(4 + 1 + 1) = sqrt(6), about 2.449
    // [docs-end:constant-spd]
    out.emplace_back("constant_spd", Json::array({ip, n, d}));
  }

  {
    // [docs-start:spd-sphere]
    Eigen::Matrix3d A = Eigen::Vector3d(4.0, 1.0, 1.0).asDiagonal();
    geodex::ConstantSPDMetric<3> weighted{A};

    geodex::Sphere<2, geodex::ConstantSPDMetric<3>> sphere_weighted{weighted};

    Eigen::Vector3d p{0.0, 0.0, 1.0};
    Eigen::Vector3d u{1.0, 0.0, 0.0};

    double n = sphere_weighted.norm(p, u);
    std::cout << "n = " << n << "\n";  // 2.0, the same weighting
    // [docs-end:spd-sphere]
    out.emplace_back("spd_sphere", Json(n));
  }

  {
    // [docs-start:projection-retraction]
    using ProjectionSphere = geodex::Sphere<2, geodex::SphereRoundMetric,
                                            geodex::SphereProjectionRetraction>;
    ProjectionSphere sphere;

    Eigen::Vector3d p{0.0, 0.0, 1.0};
    Eigen::Vector3d q{1.0, 0.0, 0.0};
    Eigen::IOFormat fmt(4, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]");

    // log and exp under the projection retraction give an approximate round trip
    Eigen::Vector3d v = sphere.log(p, q);
    Eigen::Vector3d q_approx = sphere.exp(p, v);
    std::cout << "q_approx = " << q_approx.transpose().format(fmt) << "\n";
    // [docs-end:projection-retraction]
    out.emplace_back("projection_retraction",
                     Json::array({Json(v), Json(q_approx)}));
  }

  {
    // [docs-start:se2-euler]
    // SE(2) with the Euler retraction, the cheapest one. It matches the group
    // exponential to first order only at heading 0.
    using SE2Euler =
        geodex::SE2<geodex::SE2LeftInvariantMetric, geodex::SE2EulerRetraction>;
    SE2Euler se2_euler;
    // [docs-end:se2-euler]
    out.emplace_back("se2_euler", Json(se2_euler.dim()));
  }

  geodex_examples::write_json_arg(argc, argv, Json::object(out));
  return 0;
}
