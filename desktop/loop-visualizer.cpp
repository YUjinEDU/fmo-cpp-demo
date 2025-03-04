// loop-visualizer.cpp
// 이 파일은 FMO(빠르게 움직이는 객체) 검출 시스템의 시각화 기능을 구현합니다.
// 다양한 디버그, 데모, 배경 제거 모드를 지원하여 검출 결과와 상태 정보를 실시간으로 화면에 출력합니다.

#include "desktop-opencv.hpp"
#include "loop.hpp"
#include "recorder.hpp"
#include <algorithm>
#include <fmo/processing.hpp>
#include <fmo/region.hpp>
#include <iostream>
#include <boost/range/adaptor/reversed.hpp>

// -----------------------------------------------------------------------------
// DebugVisualizer 클래스: 디버그용 시각화 처리
// -----------------------------------------------------------------------------
DebugVisualizer::DebugVisualizer(Status& s) : mStats(60) {
    // 15초 간격으로 FPS 통계 초기화
    mStats.reset(15.f);
    // 하단 도움말 메시지를 설정하는 코드는 현재 주석 처리됨
    // s.window.setBottomLine("[esc] quit | [space] pause | [enter] step | [,][.] jump 10 frames");
}

// process 메서드: 매 프레임마다 디버그 정보를 갱신하고, 검출 결과 및 평가 결과를 시각화함
void DebugVisualizer::process(Status& s, const fmo::Region& frame, const Evaluator* evaluator,
                                   const EvalResult& evalResult, fmo::Algorithm& algorithm) {
    // FPS 측정을 위해 통계 객체 업데이트
    mStats.tick();
    // 이전 FPS 값을 가져오는 코드는 주석 처리됨
    // float fpsLast = mStats.getLastFps();

    // 알고리즘이 생성한 디버그 이미지를 가져와 mVis(시각화 이미지)에 복사
    fmo::copy(algorithm.getDebugImage(mLevel, mShowIM, mShowLM, mAdd), mVis);
    // 파일 이름, 프레임 번호, FPS 등의 정보를 화면에 출력하는 코드는 주석 처리됨
    // s.window.print(s.inputName);
    // s.window.print("frame: " + std::to_string(s.inFrameNum));
    // s.window.print("fps: " + std::to_string(fpsLast));

    // 알고리즘으로부터 검출된 객체들의 결과를 가져옴
    algorithm.getOutput(mOutputCache, false);
    mObjectPoints.clear();
    // 각 검출 객체의 픽셀 좌표를 추출하여 저장
    for (auto& detection : mOutputCache.detections) {
        mObjectPoints.emplace_back();
        detection->getPoints(mObjectPoints.back());
    }
    // 현재 프레임에서 검출된 객체의 수 저장
    mDetections = mOutputCache.detections.size();

    // evaluator(평가 객체)가 제공되는 경우: 검출 결과와 정답(ground truth)을 비교하여 화면에 표시
    if (evaluator != nullptr) {
        // 평가 결과 문자열을 출력
        s.window.print(evalResult.str());
        // 현재 프레임의 ground truth 정보를 가져옴
        auto& gt = evaluator->gt().get(s.outFrameNum);
        // 검출된 점들과 정답 점들을 병합하여 각각 캐시에 저장
        fmo::pointSetMerge(begin(mObjectPoints), end(mObjectPoints), mPointsCache);
        fmo::pointSetMerge(begin(gt), end(gt), mGtPointsCache);
        // 두 점 집합을 비교하며 시각화 이미지에 그리기 (예, 검출과 실제 위치 차이를 표시)
        drawPointsGt(mPointsCache, mGtPointsCache, mVis);
        // 평가 결과가 좋은 경우 초록색, 그렇지 않으면 빨간색 텍스트로 설정
        s.window.setTextColor(good(evalResult.eval) ? Colour::green() : Colour::red());
    } else {
        // evaluator가 없는 경우, 검출된 점들을 단순히 연한 자홍색으로 그리기
        drawPoints(mPointsCache, mVis, Colour::lightMagenta());
    }
}

// processKeyboard 메서드: 사용자 키보드 입력을 처리하여 인터랙션 및 모드 전환 관리
void DebugVisualizer::processKeyboard(Status& s, const fmo::Region& frame) {
    bool step = false; // 한 프레임씩 진행할지 여부
    do {
        // 현재 일시 정지 상태에 따라 키 입력을 받아옴
        auto command = s.window.getCommand(s.paused);
        if (command == Command::PAUSE) 
            s.paused = !s.paused;  // 일시정지/재개 토글
        if (command == Command::PAUSE_FIRST) 
            mPauseFirst = !mPauseFirst;  // 첫 검출 시 일시 정지 여부 토글
        // mPauseFirst 옵션이 활성화되어 있고, 현재와 이전 검출 수가 동일하면 일시정지 토글
        if (mPauseFirst && mDetections > 0 && mPreviousDet == mDetections) {
            s.paused = !s.paused;
            mPreviousDet = 0;
            mDetections = 0;
        }
        if (command == Command::STEP) 
            step = true;  // 한 프레임씩 진행
        if (command == Command::QUIT) 
            s.quit = true;  // 프로그램 종료
        if (command == Command::SCREENSHOT) 
            fmo::save(mVis, "screenshot.png");  // 현재 시각화 이미지를 스크린샷으로 저장
        // 디버그 레벨을 0~5까지 변경하는 명령 처리
        if (command == Command::LEVEL0) mLevel = 0;
        if (command == Command::LEVEL1) mLevel = 1;
        if (command == Command::LEVEL2) mLevel = 2;
        if (command == Command::LEVEL3) mLevel = 3;
        if (command == Command::LEVEL4) mLevel = 4;
        if (command == Command::LEVEL5) mLevel = 5;
        // 이미지 출력 옵션(원본 이미지, 지역 최대값 등)을 토글 처리
        if (command == Command::SHOW_IM) mShowIM = !mShowIM;
        if (command == Command::LOCAL_MAXIMA) mShowLM = !mShowLM;
        if (command == Command::SHOW_NONE) mAdd = 0;
        if (command == Command::DIFF) mAdd = 1;
        if (command == Command::BIN_DIFF) mAdd = 2;
        if (command == Command::DIST_TRAN) mAdd = 3;

        // 카메라 입력이 없는 경우(비디오 파일 등) 프레임 점프 기능 처리
        if (!s.haveCamera()) {
            if (command == Command::JUMP_BACKWARD) {
                s.paused = false;
                s.args.frame = std::max(1, s.inFrameNum - 10);  // 10 프레임 뒤로 이동
                s.reload = true;
            }
            if (command == Command::JUMP_FORWARD) {
                s.paused = false;
                s.args.frame = s.inFrameNum + 10;  // 10 프레임 앞으로 이동
            }
        }
    } while (s.paused && !step && !s.quit);
    // 현재 검출 수를 이전 검출 수로 업데이트
    mPreviousDet = mDetections;
}

// visualize 메서드: 프레임 처리와 키보드 입력 처리를 결합하여 전체 시각화 루프를 구성함
void DebugVisualizer::visualize(Status& s, const fmo::Region& frame, const Evaluator* evaluator,
                                const EvalResult& evalResult, fmo::Algorithm& algorithm) {
    // 디버그 정보 갱신
    this->process(s, frame, evaluator, evalResult, algorithm);
    // 갱신된 시각화 이미지를 화면에 출력
    s.window.display(mVis);
    // 사용자 키 입력 처리
    this->processKeyboard(s, frame);
}

// -----------------------------------------------------------------------------
// UTIADemoVisualizer 클래스: 데모 환경에서 사용되는 시각화 (UTIA 데모)
// -----------------------------------------------------------------------------
UTIADemoVisualizer::UTIADemoVisualizer(Status &s) : vis1(s) {
    // 상단에 데모 제목 출력
    s.window.setTopLine("Fast Moving Objects Detection");
    // 디버그 시각화 객체의 모드를 2로 설정
    this->vis1.mode = 2;
}

void UTIADemoVisualizer::visualize(Status& s, const fmo::Region& frame, const Evaluator* evaluator,
                                  const EvalResult& evalResult, fmo::Algorithm& algorithm) {
    // 내부 디버그 시각화 처리 호출
    this->vis1.process(s, frame, algorithm);
    // 마지막 검출 이미지와 최대 검출 이미지가 아직 초기화되지 않았다면 현재 프레임으로 초기화
    if(mLastDetectedImage.data() == nullptr) 
        fmo::copy(frame, mLastDetectedImage);
    if(mMaxDetectedImage.data() == nullptr) 
        fmo::copy(frame, mMaxDetectedImage);
    // 이전 검출과 현재 검출 수가 모두 0보다 크면, 마지막 검출 이미지를 갱신
    if (this->mPreviousDetections * this->vis1.mNumberDetections > 0) {
        fmo::copy(this->vis1.mVis, mLastDetectedImage);
    }
    // 최대 검출 시점이 초기화되었으면 최대 검출 이미지 업데이트 및 오프셋 초기화
    if(this->vis1.mOffsetFromMaxDetection == 0) {
        fmo::copy(this->vis1.mVis, mMaxDetectedImage);
        this->mOffsetFromMax = 0;
    } else {
        this->mOffsetFromMax++;
    }

    // 모드가 2인 경우, 시각화 테이블(예: 플레이어 기록)을 활성화
    if(this->vis1.mode == 2) 
        s.window.visTable = true;
    else 
        s.window.visTable = false;
    
    // 플레이어 이름 길이를 10글자로 맞추는 처리
    auto &playerName = s.inputStrin
