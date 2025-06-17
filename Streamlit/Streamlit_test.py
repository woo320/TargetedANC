import streamlit as st
import streamlit.components.v1 as components
import tempfile
import numpy as np
import soundfile as sf
import base64
import os

st.set_page_config(layout="wide")

st.markdown("""
<style>
.wrapper-grid {
    display: flex;
    width: 100%;
    min-height: 600px;
    height: auto;
    margin: 0 auto;
    padding: 15px;
    border: 2px solid #444;
    border-radius: 12px;
    box-sizing: border-box;
    background: #f8f9fa;
    gap: 15px;
}

.left-column {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 15px;
}

.right-column {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 15px;
}

.input { 
    flex: 1;
    border: 2px solid #ff6b35;
    border-radius: 12px;
    padding: 20px;
    background: white;
    display: flex;
    flex-direction: column;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.anti { 
    flex: 1;
    border: 2px solid #2196F3;
    border-radius: 12px;
    padding: 20px;
    background: white;
    position: relative;
    display: flex;
    flex-direction: column;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.result { 
    flex: 1;
    border: 2px solid #4CAF50;
    border-radius: 12px;
    padding: 20px;
    background: white;
    position: relative;
    display: flex;
    flex-direction: column;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.controls-row {
    flex: 1;
    display: flex;
    width: 100%;
    gap: 15px;
}

.db-section {
    flex: 1;
    border: 2px solid #333;
    border-radius: 12px;
    padding: 20px;
    background: white;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.anc-section {
    flex: 1;
    border: 2px solid #333;
    border-radius: 12px;
    padding: 20px;
    background: white;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 20px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.db-display {
    font-size: 36px;
    font-weight: bold;
    color: #333;
}

canvas {
    width: 100%;
    height: 120px;
    border: 1px solid #ddd;
    border-radius: 8px;
    margin-top: 8px;
    flex-grow: 1;
    max-height: 200px;
}

video {
    width: 100%;
    height: 200px;
    border: 1px solid #ddd;
    border-radius: 8px;
    margin-top: 8px;
    object-fit: cover;
}

.label {
    font-weight: bold;
    margin-bottom: 12px;
    font-size: 18px;
    color: #333;
    text-align: center;
    padding: 8px;
    border-radius: 6px;
}

.input .label {
    background: #fff3f0;
    color: #ff6b35;
    border: 1px solid #ff6b35;
}

.anti .label {
    background: #f0f8ff;
    color: #2196F3;
    border: 1px solid #2196F3;
}

.result .label {
    background: #f0fff0;
    color: #4CAF50;
    border: 1px solid #4CAF50;
}

.toggle-switch {
    position: relative;
    width: 80px;
    height: 40px;
    background: #ddd;
    border-radius: 20px;
    cursor: pointer;
    transition: background 0.3s;
    border: 2px solid #ccc;
}

.toggle-switch.on {
    background: #4CAF50;
    border-color: #45a049;
}

.toggle-slider {
    position: absolute;
    top: 3px;
    left: 3px;
    width: 32px;
    height: 32px;
    background: white;
    border-radius: 50%;
    transition: transform 0.3s;
    box-shadow: 0 2px 6px rgba(0,0,0,0.3);
}

.toggle-switch.on .toggle-slider {
    transform: translateX(40px);
}

.anc-label {
    font-size: 20px;
    font-weight: bold;
    color: #333;
}

.upload-section {
    display: flex;
    gap: 20px;
    margin-bottom: 20px;
}

.upload-box {
    flex: 1;
    padding: 20px;
    border: 2px dashed #ccc;
    border-radius: 12px;
    text-align: center;
    background: #f9f9f9;
}

.play-button {
    background: #4CAF50;
    color: white;
    border: none;
    border-radius: 50%;
    width: 60px;
    height: 60px;
    font-size: 24px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 10px auto;
    transition: background 0.3s;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

.play-button:hover {
    background: #45a049;
}

.play-button.pause {
    background: #f44336;
}

.play-button.pause:hover {
    background: #da190b;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center; color:#333; margin-bottom:30px;'>🎧 Targeted ANC</h1>", unsafe_allow_html=True)

# 파일 경로 설정
video_file_path = "창신역.mp4"
noise_file_path = "창신역_Mix.wav"
anti_file_path = "창신역_Final.wav"

# 파일 존재 확인
if os.path.exists(video_file_path) and os.path.exists(noise_file_path) and os.path.exists(anti_file_path):
    # 오디오 파일 처리 - 시간 길이만 짧은 것에 맞춤
    noise_data, noise_sr = sf.read(noise_file_path)
    if len(noise_data.shape) > 1:
        noise_data = noise_data[:, 0]
    
    anti_data, anti_sr = sf.read(anti_file_path)
    if len(anti_data.shape) > 1:
        anti_data = anti_data[:, 0]

    # 정규화
    noise_data = noise_data / np.max(np.abs(noise_data)) if np.max(np.abs(noise_data)) > 0 else noise_data
    anti_data = anti_data / np.max(np.abs(anti_data)) if np.max(np.abs(anti_data)) > 0 else anti_data

    # 시간 길이를 짧은 것에 맞춤 (샘플레이트는 원본 유지)
    noise_duration = len(noise_data) / noise_sr
    anti_duration = len(anti_data) / anti_sr
    min_duration = min(noise_duration, anti_duration)
    
    # 샘플 수를 시간에 맞춰 조정
    noise_samples = int(min_duration * noise_sr)
    anti_samples = int(min_duration * anti_sr)
    
    noise_data = noise_data[:noise_samples]
    anti_data = anti_data[:anti_samples]
    
    print(f"Noise: {len(noise_data)} samples, {noise_sr}Hz, {len(noise_data)/noise_sr:.2f}s")
    print(f"Anti: {len(anti_data)} samples, {anti_sr}Hz, {len(anti_data)/anti_sr:.2f}s")

    # 임시 파일 생성 - 각각 원본 샘플레이트 유지
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_noise:
        sf.write(tmp_noise.name, noise_data, noise_sr, subtype='PCM_16')
        noise_path = tmp_noise.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_anti:
        sf.write(tmp_anti.name, anti_data, anti_sr, subtype='PCM_16')
        anti_path = tmp_anti.name

    # 비디오 파일 처리
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
        with open(video_file_path, "rb") as f:
            tmp_video.write(f.read())
        video_path = tmp_video.name

    # Base64 인코딩
    with open(noise_path, "rb") as f:
        noise_b64 = base64.b64encode(f.read()).decode("utf-8")
    
    with open(anti_path, "rb") as f:
        anti_b64 = base64.b64encode(f.read()).decode("utf-8")
    
    # Result는 JavaScript에서 계산하므로 빈 값으로 설정
    result_b64 = ""
    
    with open(video_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode("utf-8")

    html_string = """
    <div style="display: flex; width: 100%; height: 700px; border: 2px solid #444; border-radius: 12px; background: #f8f9fa; padding: 15px; box-sizing: border-box; gap: 15px;">
        <!-- Left Column -->
        <div style="flex: 1; display: flex; flex-direction: column; gap: 15px;">
            <!-- Video Section - 2/3 비율 -->
            <div style="flex: 3; min-height: 0; border: 2px solid #ff6b35; border-radius: 12px; padding: 20px; background: white; display: flex; flex-direction: column; box-shadow: 0 4px 8px rgba(0,0,0,0.1); position: relative;">
                <div style="font-weight: bold; margin-bottom: 12px; font-size: 18px; color: #ff6b35; text-align: center; padding: 8px; border-radius: 6px; background: #fff3f0; border: 1px solid #ff6b35;">Input Video</div>
                <video id="inputVideo" muted preload="auto" style="width: 100%; height: 100%; border: 1px solid #ddd; border-radius: 8px; object-fit: cover; flex-grow: 1; min-height: 0;" src="data:video/{video_ext};base64,{video_b64}" crossorigin="anonymous"></video>
            </div>
            
            <!-- Control Section - 1/3 비율 -->
            <div style="flex: 1; display: flex; gap: 15px;">
                <!-- Play Button Section -->
                <div style="flex: 1; border: 2px solid #333; border-radius: 12px; padding: 20px; background: white; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                    <button id="masterPlayButton" onclick="togglePlayPause()" style="background: #4CAF50; color: white; border: none; border-radius: 50%; width: 60px; height: 60px; font-size: 24px; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: background 0.3s; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
                        ▶
                    </button>
                </div>
                
                <!-- dB SPL Display Section -->
                <div style="flex: 1; border: 2px solid #333; border-radius: 12px; padding: 20px; background: white; display: flex; flex-direction: column; align-items: center; justify-content: center; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                    <div style="font-size: 36px; font-weight: bold; color: #333;"><span id="dbValue">0</span>dB</div>
                </div>
                
                <!-- ANC Toggle Section -->
                <div style="flex: 1; border: 2px solid #333; border-radius: 12px; padding: 20px; background: white; display: flex; align-items: center; justify-content: center; gap: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                    <div style="font-size: 20px; font-weight: bold; color: #333;">ANC</div>
                    <div id="toggleSwitch" onclick="toggleANC()" style="position: relative; width: 80px; height: 40px; background: #ddd; border-radius: 20px; cursor: pointer; transition: background 0.3s; border: 2px solid #ccc;">
                        <div style="position: absolute; top: 3px; left: 3px; width: 32px; height: 32px; background: white; border-radius: 50%; transition: transform 0.3s; box-shadow: 0 2px 6px rgba(0,0,0,0.3);"></div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Right Column -->
        <div style="flex: 1; display: flex; flex-direction: column; gap: 15px;">
            <!-- Input Noise Section -->
            <div style="flex: 1; border: 2px solid #2196F3; border-radius: 12px; padding: 20px; background: white; display: flex; flex-direction: column; box-shadow: 0 4px 8px rgba(0,0,0,0.1); position: relative;">
                <div style="font-weight: bold; margin-bottom: 12px; font-size: 18px; color: #2196F3; text-align: center; padding: 8px; border-radius: 6px; background: #f0f8ff; border: 1px solid #2196F3;">Input Noise</div>
                <canvas id="noiseCanvas" style="width: 100%; height: 120px; border: 1px solid #ddd; border-radius: 8px; margin-top: 8px; flex-grow: 1; max-height: 150px;"></canvas>
            </div>
            
            <!-- Result Section -->
            <div style="flex: 1; border: 2px solid #4CAF50; border-radius: 12px; padding: 20px; background: white; display: flex; flex-direction: column; box-shadow: 0 4px 8px rgba(0,0,0,0.1); position: relative;">
                <div style="font-weight: bold; margin-bottom: 12px; font-size: 18px; color: #4CAF50; text-align: center; padding: 8px; border-radius: 6px; background: #f0fff0; border: 1px solid #4CAF50;">Result</div>
                <canvas id="resultCanvas" style="width: 100%; height: 120px; border: 1px solid #ddd; border-radius: 8px; margin-top: 8px; flex-grow: 1; max-height: 150px;"></canvas>
            </div>
        </div>
    </div>

    <!-- Hidden audio elements -->
    <audio id="noisePlayer" preload="auto" style="display: none;" crossorigin="anonymous">
        <source src="data:audio/wav;base64,{noise_b64}" type="audio/wav">
        Your browser does not support the audio element.
    </audio>
    <audio id="antiPlayer" preload="auto" style="display: none;" crossorigin="anonymous">
        <source src="data:audio/wav;base64,{anti_b64}" type="audio/wav">
        Your browser does not support the audio element.
    </audio>

    <script>
    // 글로벌 변수들
    let video, noisePlayer, antiPlayer;
    let ctxNoise, ctxResult;
    let dbDisplay, toggleSwitch, masterPlayButton;
    let audioCtx, sourceNodeNoise, sourceNodeAnti, analyserNoise, analyserAnti, analyserCombined;
    let gainNodeNoise, gainNodeAnti;
    let ancEnabled = false;
    let isPlaying = false;
    let animationId;

    // DOM 요소들 초기화
    function initElements() {{
        video = document.getElementById("inputVideo");
        noisePlayer = document.getElementById("noisePlayer");
        antiPlayer = document.getElementById("antiPlayer");
        ctxNoise = document.getElementById("noiseCanvas").getContext("2d");
        ctxResult = document.getElementById("resultCanvas").getContext("2d");
        dbDisplay = document.getElementById("dbValue");
        toggleSwitch = document.getElementById("toggleSwitch");
        masterPlayButton = document.getElementById("masterPlayButton");
        
        console.log('Elements initialized');
    }}

    function setup() {{
        try {{
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            
            sourceNodeNoise = audioCtx.createMediaElementSource(noisePlayer);
            sourceNodeAnti = audioCtx.createMediaElementSource(antiPlayer);
            
            gainNodeNoise = audioCtx.createGain();
            gainNodeAnti = audioCtx.createGain();
            
            analyserNoise = audioCtx.createAnalyser();
            analyserAnti = audioCtx.createAnalyser();
            analyserCombined = audioCtx.createAnalyser();

            sourceNodeNoise.connect(gainNodeNoise);
            sourceNodeAnti.connect(gainNodeAnti);
            
            gainNodeNoise.connect(analyserNoise);
            gainNodeAnti.connect(analyserAnti);
            
            gainNodeNoise.connect(analyserCombined);
            gainNodeAnti.connect(analyserCombined);
            
            gainNodeNoise.connect(audioCtx.destination);
            gainNodeAnti.connect(audioCtx.destination);

            analyserNoise.fftSize = 2048;
            analyserAnti.fftSize = 2048;
            analyserCombined.fftSize = 2048;
            
            // 초기 상태: ANC OFF
            gainNodeNoise.gain.value = 1.0;
            gainNodeAnti.gain.value = 0.0;
            
            console.log('Audio setup complete');
            animate();
        }} catch (error) {{
            console.error('Setup error:', error);
        }}
    }}

    function togglePlayPause() {{
        try {{
            if (!audioCtx) {{
                setup();
            }}
            
            if (isPlaying) {{
                // 정지
                video.pause();
                noisePlayer.pause();
                antiPlayer.pause();
                masterPlayButton.innerHTML = "▶";
                masterPlayButton.style.background = "#4CAF50";
                isPlaying = false;
                dbDisplay.textContent = "0";
                console.log('Paused');
            }} else {{
                // 재생
                if (audioCtx && audioCtx.state === 'suspended') {{
                    audioCtx.resume();
                }}
                
                // 비디오 시간에 맞춰 오디오 동기화
                noisePlayer.currentTime = video.currentTime;
                antiPlayer.currentTime = video.currentTime;
                
                video.play();
                noisePlayer.play();
                antiPlayer.play();
                
                masterPlayButton.innerHTML = "⏸";
                masterPlayButton.style.background = "#f44336";
                isPlaying = true;
                console.log('Playing');
            }}
        }} catch (error) {{
            console.error('Play/Pause error:', error);
        }}
    }}

    function toggleANC() {{
        ancEnabled = !ancEnabled;
        const slider = toggleSwitch.querySelector('div');
        
        console.log('ANC toggled to:', ancEnabled);
        
        if (ancEnabled) {{
            toggleSwitch.style.background = "#4CAF50";
            toggleSwitch.style.borderColor = "#45a049";
            slider.style.transform = "translateX(40px)";
            
            if (gainNodeNoise && gainNodeAnti) {{
                gainNodeNoise.gain.value = 0.0;
                gainNodeAnti.gain.value = 1.0;
            }}
        }} else {{
            toggleSwitch.style.background = "#ddd";
            toggleSwitch.style.borderColor = "#ccc";
            slider.style.transform = "translateX(0px)";
            
            if (gainNodeNoise && gainNodeAnti) {{
                gainNodeNoise.gain.value = 1.0;
                gainNodeAnti.gain.value = 0.0;
            }}
        }}
    }}

    function drawCenterLine(ctx) {{
        ctx.strokeStyle = "#e0e0e0";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, ctx.canvas.height / 2);
        ctx.lineTo(ctx.canvas.width, ctx.canvas.height / 2);
        ctx.stroke();
    }}

    function drawWaveform(ctx, dataArray, color) {{
        ctx.lineWidth = 3;
        ctx.strokeStyle = color;
        ctx.beginPath();
        const sliceWidth = ctx.canvas.width / dataArray.length;
        let x = 0;
        for (let i = 0; i < dataArray.length; i++) {{
            const v = dataArray[i] / 128.0;
            const y = v * ctx.canvas.height / 2;
            if (i === 0) {{
                ctx.moveTo(x, y);
            }} else {{
                ctx.lineTo(x, y);
            }}
            x += sliceWidth;
        }}
        ctx.stroke();
    }}

    function calculateDBSPL(timeData) {{
        let sumSquares = 0;
        const dataLength = timeData.length;
        
        for (let i = 0; i < dataLength; i++) {{
            const normalizedAmplitude = (timeData[i] - 128) / 128.0;
            sumSquares += normalizedAmplitude * normalizedAmplitude;
        }}
        
        const rms = Math.sqrt(sumSquares / dataLength);
        
        if (rms > 0) {{
            const dbSPL = 20 * Math.log10(rms) + 94;
            return Math.max(0, Math.round(dbSPL * 10) / 10);
        }} else {{
            return 0;
        }}
    }}

    function animate() {{
        animationId = requestAnimationFrame(animate);
        
        if (!analyserNoise || !analyserAnti || !analyserCombined) return;
        
        try {{
            const timeDataNoise = new Uint8Array(analyserNoise.fftSize);
            const timeDataAnti = new Uint8Array(analyserAnti.fftSize);
            const timeDataCombined = new Uint8Array(analyserCombined.fftSize);
            
            analyserNoise.getByteTimeDomainData(timeDataNoise);
            analyserAnti.getByteTimeDomainData(timeDataAnti);
            analyserCombined.getByteTimeDomainData(timeDataCombined);

            // 캔버스 완전히 비우기
            ctxNoise.clearRect(0, 0, ctxNoise.canvas.width, ctxNoise.canvas.height);
            ctxResult.clearRect(0, 0, ctxResult.canvas.width, ctxResult.canvas.height);
            
            // 중앙선 그리기
            drawCenterLine(ctxNoise);
            drawCenterLine(ctxResult);

            // ANC 상태에 따른 파형 표시
            if (ancEnabled) {{
                // ANC ON: Result에만 파형
                drawWaveform(ctxResult, timeDataCombined, "#4CAF50");
            }} else {{
                // ANC OFF: Input Noise에만 파형
                drawWaveform(ctxNoise, timeDataNoise, "#2196F3");
            }}

            // dB 표시
            if (isPlaying) {{
                const currentDBSPL = calculateDBSPL(timeDataCombined);
                dbDisplay.textContent = currentDBSPL;
            }}
        }} catch (error) {{
            console.error('Animation error:', error);
        }}
    }}

    // 이벤트 리스너 설정 - ended 이벤트 제거
    function setupEventListeners() {{
        // 비디오와 오디오의 ended 이벤트를 제거하여 10초 문제 해결
        // 대신 timeupdate로 동기화만 처리
        
        video.addEventListener('timeupdate', () => {{
            if (isPlaying) {{
                const videoTime = video.currentTime;
                if (Math.abs(noisePlayer.currentTime - videoTime) > 0.1) {{
                    noisePlayer.currentTime = videoTime;
                }}
                if (Math.abs(antiPlayer.currentTime - videoTime) > 0.1) {{
                    antiPlayer.currentTime = videoTime;
                }}
            }}
        }});
        
        video.addEventListener('loadeddata', () => {{
            console.log('Video loaded, duration:', video.duration);
        }});
        
        noisePlayer.addEventListener('loadeddata', () => {{
            console.log('Noise audio loaded, duration:', noisePlayer.duration);
        }});
        
        antiPlayer.addEventListener('loadeddata', () => {{
            console.log('Anti-noise audio loaded, duration:', antiPlayer.duration);
        }});
    }}

    // 초기화 실행
    document.addEventListener('DOMContentLoaded', function() {{
        initElements();
        setupEventListeners();
        console.log('App initialized');
    }});

    // 페이지가 이미 로드된 경우를 위한 즉시 실행
    if (document.readyState === 'loading') {{
        document.addEventListener('DOMContentLoaded', function() {{
            initElements();
            setupEventListeners();
        }});
    }} else {{
        initElements();
        setupEventListeners();
    }}
    </script>
    """.format(
        video_ext=os.path.splitext(video_file_path)[1][1:],
        video_b64=video_b64,
        noise_b64=noise_b64,
        anti_b64=anti_b64
    )

    components.html(html_string, height=750)

    # 파일 정리
    try:
        os.unlink(noise_path)
        os.unlink(anti_path)
        os.unlink(video_path)
    except:
        pass

else:
    st.error("📁 필요한 파일들이 없습니다!")
    st.info("현재 폴더에 다음 파일들을 준비해주세요:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**창신역.mp4**")
    with col2:
        st.markdown("**창신역_Mix.wav**")
    with col3:
        st.markdown("**창신역_Final.wav**")