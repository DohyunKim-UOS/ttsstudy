import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# ===== 설정 =====
wav_path = "dtmf_12345.wav"  # 분석할 WAV 파일 경로
out_png  = "spectrogram_dtmf_scale.png"     # 저장 파일명

# STFT 파라미터 (DTMF에 무난한 설정)
sr_target = None   # None이면 원본 SR 사용(전화음은 보통 8000 Hz)
n_fft = 1024
hop_length = 256
win = "hann"
center = True

# DTMF 표준 주파수 (Hz)
dtmf_low  = [697, 770, 852, 941]
dtmf_high = [1209, 1336, 1477, 1633]
dtmf_all  = dtmf_low + dtmf_high

# ===== 로드 =====
y, sr = librosa.load(wav_path, sr=sr_target, mono=True)

# ===== STFT & dB 변환 =====
S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=win, center=center)
S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

# ===== 표시: DTMF 주파수 스케일에 맞춘 축 =====
fig, ax = plt.subplots(figsize=(9, 4.8))
img = librosa.display.specshow(
    S_db,
    sr=sr,
    hop_length=hop_length,
    x_axis="time",
    y_axis="linear",    # DTMF는 선형 축이 직관적
    ax=ax
)

# y축 범위를 DTMF 대역에 맞춰 제한(필요 시 조정)
ax.set_ylim(500, 1800)

# DTMF 주파수 눈금 및 보조선
yticks = sorted(dtmf_all)
ax.set_yticks(yticks)
ax.set_yticklabels([str(f) for f in yticks])
# 수평 가이드라인(점선)
for f in yticks:
    ax.axhline(f, linestyle="--", linewidth=0.6, alpha=0.6)

# 격자 및 컬러바
ax.grid(True, which="both", axis="both", linestyle=":", linewidth=0.5, alpha=0.6)
cbar = plt.colorbar(img, ax=ax, format="%+2.0f dB")
cbar.set_label("Magnitude (dB)")

# 주석: 저역/고역 대역
ax.text(0.99, (dtmf_low[-1]-500)/(1800-500), "Low group\n(697–941 Hz)",
        ha="right", va="bottom", transform=ax.transAxes)
ax.text(0.99, (dtmf_high[0]-500)/(1800-500), "High group\n(1209–1633 Hz)",
        ha="right", va="bottom", transform=ax.transAxes)

ax.set_title(f"DTMF-aligned Spectrogram: {os.path.basename(wav_path)}")
plt.tight_layout()
plt.savefig(out_png, dpi=150)
plt.show()

print(f"Saved spectrogram to: {out_png}")
