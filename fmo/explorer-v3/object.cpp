#include "../include-opencv.hpp" 
#include "explorer.hpp"
#include <algorithm>
#include <fmo/processing.hpp>
#include <fmo/region.hpp>
#include <limits>
#include <type_traits>

namespace fmo {

    // 내부 익명 네임스페이스에서 경계값 상수를 정의 (16비트 정수의 최소/최대값)
    namespace {
        constexpr int BOUNDS_MIN = std::numeric_limits<int16_t>::min();
        constexpr int BOUNDS_MAX = std::numeric_limits<int16_t>::max();
    }

    // ExplorerV3 클래스: 이미지에서 클러스터(스트립들의 그룹)를 분석하여 객체(공 등)를 찾아내는 알고리즘 구현
    void ExplorerV3::findObjects() {
        // 이전에 저장된 객체 후보들을 모두 초기화
        mObjects.clear();

        // 클러스터를 길이(total length)에 따라 내림차순으로 정렬하기 위해 임시 벡터 사용
        auto& sortClusters = mCache.sortClusters;
        sortClusters.clear();

        // 모든 클러스터를 순회하면서, 유효한(Invalid 상태가 아닌) 클러스터만 추가
        for (auto& cluster : mClusters) {
            if (cluster.isInvalid()) continue;
            sortClusters.emplace_back(cluster.lengthTotal, &cluster);
        }

        // 내림차순 정렬: 길이가 긴 클러스터가 앞쪽에 오도록 정렬
        std::sort(begin(sortClusters), end(sortClusters),
                  [](auto& l, auto& r) { return l.first > r.first; });

        // 가장 긴 클러스터부터 순회하면서 객체로 판단될 수 있는지 검사
        for (auto& sortCluster : sortClusters) {
            if (isObject(*sortCluster.second)) {
                // 객체 조건을 만족하면 mObjects에 해당 클러스터를 추가하고 반복 종료
                mObjects.push_back(sortCluster.second);
                break;
            } else {
                // 객체 조건을 만족하지 않으면 클러스터를 '객체가 아님' 상태로 표시
                sortCluster.second->setInvalid(Cluster::NOT_AN_OBJECT);
            }
        }
    }

    // isObject: 주어진 클러스터가 객체(예: 공)로 인식될 수 있는지 여러 조건을 검사함
    bool ExplorerV3::isObject(Cluster& cluster) const {
        // 클러스터에 포함된 스트립들을 바탕으로 두 가지 차이 이미지(새로운/오래된)에 대한 경계박스 계산
        cluster.bounds1 = findClusterBoundsInDiff(cluster, true);
        cluster.bounds2 = findClusterBoundsInDiff(cluster, false);

        // 조건 1: 두 차이 이미지 모두에 적어도 하나 이상의 스트립이 있어야 함
        if (cluster.bounds1.min.x == BOUNDS_MAX || cluster.bounds2.min.x == BOUNDS_MAX)
            return false;

        // 강제로 왼쪽에서 오른쪽으로 진행하는 방향을 보장
        int xMin = cluster.l.pos.x;
        if (cluster.bounds1.min.x != xMin) {
            // 만약 왼쪽 가장자리 좌표가 다르면 두 경계 범위를 교환
            std::swap(cluster.bounds1, cluster.bounds2);
        }

        // 조건 2: 왼쪽 가장자리 스트립은 bounds1에 있어야 함
        if (cluster.bounds1.min.x != xMin) return false;

        // 조건 3: 오른쪽 가장자리 스트립은 bounds2에 있어야 함
        int xMax = cluster.r.pos.x;
        if (cluster.bounds2.max.x != xMax) return false;

        // 조건 4: bounds1(첫 번째 범위)의 오른쪽 끝과 오른쪽 가장자리 사이의 거리가 일정 범위 내에 있어야 함
        int minMotion = int(mCfg.minMotion * (xMax - xMin));
        int maxMotion = int(mCfg.maxMotion * (xMax - xMin));
        if (xMax - cluster.bounds1.max.x < minMotion) return false;
        if (xMax - cluster.bounds1.max.x > maxMotion) return false;

        // 조건 5: bounds2(두 번째 범위)의 왼쪽 시작점과 왼쪽 가장자리 사이의 거리가 일정 범위 내에 있어야 함
        if (cluster.bounds2.min.x - xMin < minMotion) return false;
        if (cluster.bounds2.min.x - xMin > maxMotion) return false;

        return true;
    }

    // findClusterBoundsInDiff: 클러스터 내의 스트립들을 순회하며, 차이 이미지(새로운/오래된)에 존재하는 스트립들의 바운딩 박스를 계산함
    Bounds ExplorerV3::findClusterBoundsInDiff(const Cluster& cluster, bool newer) const {
        // 초기값 설정: 최소값은 큰 수(BOUNDS_MAX), 최대값은 작은 수(BOUNDS_MIN)로 시작
        Bounds result{{BOUNDS_MAX, BOUNDS_MAX}, {BOUNDS_MIN, BOUNDS_MIN}};

        // 클러스터의 시작 스트립 인덱스부터 순회
        int index = cluster.l.strip;
        while (index != MetaStrip::END) {
            auto& strip = mLevel.metaStrips[index];

            // 조건: strip이 차이 이미지에 나타나는 경우 (newer 플래그에 따라 newer 또는 older)
            if (newer ? strip.newer : strip.older) {
                // 현재 스트립의 위치로 경계값을 업데이트
                result.min.x = std::min(result.min.x, int(strip.pos.x));
                result.min.y = std::min(result.min.y, int(strip.pos.y));
                result.max.x = std::max(result.max.x, int(strip.pos.x));
                result.max.y = std::max(result.max.y, int(strip.pos.y));
            }
            // 다음 스트립으로 이동
            index = strip.next;
        }

        return result;
    }

    // MyDetection 생성자: Detection 기본 클래스의 생성자 호출 후, ExplorerV3 인스턴스와 관련 클러스터 포인터 저장
    ExplorerV3::MyDetection::MyDetection(const Detection::Object& detObj,
                                         const Detection::Predecessor& detPrev,
                                         const Cluster* cluster, const ExplorerV3* aMe)
        : Detection(detObj, detPrev), me(aMe), mCluster(cluster) {}

    // getPoints: 검출된 객체(MyDetection)에서 실제 픽셀 좌표들을 추출하여 PointSet에 저장함
    void ExplorerV3::MyDetection::getPoints(PointSet& out) const {
        out.clear();
        auto& obj = *mCluster;

        // 클러스터 내 모든 스트립들을 순회
        int index = obj.l.strip;
        while (index != MetaStrip::END) {
            auto& strip = me->mLevel.metaStrips[index];

            // 조건: 스트립이 오래된 이미지와 새로운 이미지 둘 다에 존재하는 경우
            if (strip.older && strip.newer) {
                // 스트립의 영역(사각형 영역)의 모든 픽셀을 객체의 픽셀로 추가
                int ye = strip.pos.y + strip.halfDims.height;
                int xe = strip.pos.x + strip.halfDims.width;

                for (int y = strip.pos.y - strip.halfDims.height; y < ye; y++) {
                    for (int x = strip.pos.x - strip.halfDims.width; x < xe; x++) {
                        out.push_back({x, y});
                    }
                }
            }
            // 다음 스트립으로 이동
            index = strip.next;
        }

        // 빠른 비교를 위해 포인트 리스트 정렬 (pointSetCompLt 비교 함수 사용)
        std::sort(begin(out), end(out), pointSetCompLt);
    }

    // 익명 네임스페이스 내 유틸리티 함수들
    namespace {
        // center: 주어진 경계(Bounds)의 중심 좌표를 계산하여 반환
        Pos center(const fmo::Bounds& b) {
            return {(b.max.x + b.min.x) / 2, (b.max.y + b.min.y) / 2};
        }

        // average: 두 실수 값의 평균을 계산
        float average(float v1, float v2) { return (v1 + v2) / 2; }
    }

    // getOutput: 검출 결과를 Detection 객체들의 리스트 형태로 반환함
    void ExplorerV3::getOutput(Output &out, bool smoothTrajecotry) {
        out.clear();
        Detection::Object detObj;
        Detection::Predecessor detPrev;

        // mObjects에 저장된 객체 후보(클러스터)를 순회
        for (auto* cluster : mObjects) {
            // bounds1을 사용해 객체의 중심 좌표 계산
            detObj.center = center(cluster->bounds1);
            // 클러스터의 최소/최대 높이의 평균을 객체의 반지름(radius)로 설정
            detObj.radius = average(cluster->approxHeightMin, cluster->approxHeightMax);
            // bounds2를 사용해 전 프레임(Predecessor)의 중심 좌표 계산
            detPrev.center = center(cluster->bounds2);
            // Detection 리스트에 새 객체를 추가 (MyDetection 클래스 사용)
            out.detections.emplace_back();
            out.detections.back().reset(new MyDetection(detObj, detPrev, cluster, this));
        }
    }
}
