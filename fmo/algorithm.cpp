#include <fmo/algorithm.hpp>
#include <map>

namespace fmo {

    // Algorithm::Config 클래스 생성자
    // 검출 알고리즘의 다양한 파라미터를 초기화합니다.
    Algorithm::Config::Config()
        : name("taxonomy-v1"),          // 사용할 알고리즘 이름 (기본값: "taxonomy-v1")
          diff(),                       // 차이(diff) 관련 파라미터 (기본 생성자 호출)
          //
          iouThreshold(0.5f),           // Intersection over Union 임계값
          maxGapX(0.020f),              // X축 최대 간격
          minGapY(0.046f),              // Y축 최소 간격
          maxImageHeight(300),          // 최대 이미지 높이
          minStripHeight(2),            // 최소 스트립(연속된 픽셀 영역) 높이
          minStripsInObject(4),         // 하나의 객체로 판단하기 위한 최소 스트립 수
          minStripArea(0.43f),          // 최소 스트립 면적
          minAspect(1.0f),              // 최소 가로세로 비율
          minAspectForRelevantAngle(1.62f), // 특정 각도에서 관련 있는 최소 가로세로 비율
          minDistToTMinus2(1.9f),       // 시간 T-2와의 최소 거리
          matchAspectMax(1.57f),        // 매칭할 때 최대 가로세로 비율
          matchAreaMax(2.15f),          // 매칭할 때 최대 면적 비율
          matchDistanceMin(0.55f),      // 매칭할 때 최소 거리
          matchDistanceMax(5.f),        // 매칭할 때 최대 거리
          matchAngleMax(0.37f),         // 매칭할 때 최대 각도 차이
          matchAspectWeight(1.00f),     // 매칭 시 가로세로 비율의 가중치
          matchAreaWeight(1.35f),       // 매칭 시 면적의 가중치
          matchDistanceWeight(0.25f),   // 매칭 시 거리의 가중치
          matchAngleWeight(5.00f),      // 매칭 시 각도의 가중치
          selectMaxDistance(0.60f),     // 선택할 때 최대 거리
          outputRadiusCorr(1.f),        // 출력 시 반지름 보정 계수
          outputRadiusMin(2.f),         // 출력 시 최소 반지름
          outputRasterCorr(1.f),        // 출력 시 래스터 보정 계수
          outputNoRobustRadius(false),  // 출력 시 강인한 반지름 계산을 사용하지 않을지 여부
          //
          imageHeight(480),             // 입력 이미지의 높이
          //
          minStripsInComponent(2),      // 하나의 컴포넌트(연결된 영역)로 판단하기 위한 최소 스트립 수
          minStripsInCluster(12),       // 클러스터(객체 후보)를 구성하는 최소 스트립 수
          minClusterLength(2.f),        // 클러스터의 최소 길이
          heightRatioWeight(1.f),       // 높이 비율의 가중치
          distanceWeight(0.f),          // 거리의 가중치
          gapsWeight(1.f),              // 간격의 가중치
          maxHeightRatioStrips(1.75001f),   // 스트립 간 최대 높이 비율
          maxHeightRatioInternal(1.75001f), // 내부 스트립 간 최대 높이 비율
          maxHeightRatioExternal(1.99999f), // 외부 스트립 간 최대 높이 비율
          maxDistance(20.f),            // 최대 허용 거리
          maxGapsLength(0.75f),         // 최대 간격 길이
          minMotion(0.25f),             // 최소 움직임 임계값
          maxMotion(0.50f),             // 최대 움직임 임계값
          pointSetSourceResolution(false)  // 점 집합(point set) 해상도 사용 여부
    {}

    // 알고리즘 팩토리들을 등록하기 위한 맵(레지스트리) 정의
    using AlgorithmRegistry = std::map<std::string, Algorithm::Factory>;

    // 전역 레지스트리 반환 함수 (정적 변수로 한번만 생성됨)
    AlgorithmRegistry& getRegistry() {
        static AlgorithmRegistry registry;
        return registry;
    }

    // 아래 함수들은 각각 다른 알고리즘 팩토리를 등록하는 함수들입니다.
    void registerExplorerV1();
    void registerExplorerV2();
    void registerExplorerV3();
    void registerMedianV1();
    void registerMedianV2();
    void registerTaxonomyV1();

    // 내장된(빌트인) 팩토리들을 한번만 등록하는 함수
    void registerBuiltInFactories() {
        static bool registered = false;
        if (registered) return;
        registered = true;

        // 각 알고리즘 팩토리 등록 함수 호출
        registerExplorerV1();
        registerExplorerV2();
        registerExplorerV3();
        registerMedianV1();
        registerMedianV2();
        registerTaxonomyV1();
    }

    // Algorithm 객체를 생성하는 팩토리 메서드
    // 주어진 Config, 포맷, 이미지 크기를 사용하여 적절한 알고리즘 객체를 생성합니다.
    std::unique_ptr<Algorithm> fmo::Algorithm::make(const Config& config, Format format, Dims dims) {
        registerBuiltInFactories(); // 내장 팩토리들을 등록
        auto& registry = getRegistry();
        auto it = registry.find(config.name);  // config에 지정된 이름으로 팩토리를 찾음
        if (it == registry.end()) { 
            throw std::runtime_error("unknown algorithm name");  // 등록된 이름이 아니면 예외 발생
        }
        return it->second(config, format, dims);  // 팩토리를 호출하여 객체 생성 후 반환
    }

    // 새로운 알고리즘 팩토리를 등록하는 함수
    void Algorithm::registerFactory(const std::string& name, const Factory& factory) {
        auto& registry = getRegistry();
        auto it = registry.find(name);
        if (it != registry.end()) { 
            throw std::runtime_error("duplicate algorithm name");  // 이미 등록된 이름이면 예외 발생
        }
        registry.emplace(name, factory);  // 새로운 팩토리 등록
    }

    // 현재 등록된 알고리즘 팩토리들의 이름 목록을 반환하는 함수
    std::vector<std::string> Algorithm::listFactories() {
        registerBuiltInFactories(); // 내장 팩토리들을 등록
        const auto& registry = getRegistry();
        std::vector<std::string> result;

        // 레지스트리에 등록된 각 항목의 이름을 결과 벡터에 추가
        for (auto& entry : registry) { 
            result.push_back(entry.first); 
        }

        return result;
    }
}
