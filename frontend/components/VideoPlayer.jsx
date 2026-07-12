"use client";

import { MediaPlayer, MediaProvider, Track } from "@vidstack/react";
import { defaultLayoutIcons, DefaultVideoLayout } from "@vidstack/react/player/layouts/default";
import { parseText as loadMediaCaptions } from "media-captions";
import "@vidstack/react/player/styles/default/theme.css";
import "@vidstack/react/player/styles/default/layouts/video.css";

void loadMediaCaptions;

const PLAYER_TRANSLATIONS = {
  AirPlay: "AirPlay",
  Accessibility: "Hiển thị",
  Audio: "Âm lượng",
  Auto: "Tự động",
  Boost: "Tăng cường",
  Captions: "Phụ đề",
  "Caption Styles": "Kiểu phụ đề",
  "Captions look like this": "Phụ đề sẽ hiển thị như thế này",
  Chapters: "Chương",
  "Closed-Captions Off": "Tắt phụ đề",
  "Closed-Captions On": "Bật phụ đề",
  Color: "Màu",
  Default: "Mặc định",
  Disabled: "Tắt",
  "Display Background": "Nền hiển thị",
  Download: "Tải xuống",
  "Enter Fullscreen": "Toàn màn hình",
  "Enter PiP": "Mở PiP",
  "Exit Fullscreen": "Thoát toàn màn hình",
  "Exit PiP": "Thoát PiP",
  Family: "Font",
  Font: "Font",
  Fullscreen: "Toàn màn hình",
  Loop: "Lặp lại",
  Mute: "Tắt tiếng",
  Normal: "Bình thường",
  Off: "Tắt",
  Opacity: "Độ mờ",
  Pause: "Tạm dừng",
  PiP: "PiP",
  Play: "Phát",
  Playback: "Tốc độ",
  Quality: "Chất lượng",
  Replay: "Phát lại",
  Reset: "Đặt lại",
  "Seek Backward": "Tua lại",
  "Seek Forward": "Tua tới",
  Seek: "Tua",
  Settings: "Cài đặt",
  Shadow: "Đổ bóng",
  Size: "Cỡ chữ",
  Speed: "Tốc độ",
  Text: "Chữ",
  "Text Background": "Nền chữ",
  Track: "Track",
  Unmute: "Bật tiếng",
  Volume: "Âm lượng",
};

export default function VideoPlayer({
  src,
  title,
  poster,
  playerRef,
  onTime,
  subtitles,
  chapters,
}) {
  return (
    <MediaPlayer
      ref={playerRef}
      title={title}
      src={{ src, type: "video/mp4" }}
      poster={poster}
      playsInline
      aspectRatio="16/9"
      onTimeUpdate={(d) => onTime?.(d.currentTime)}
      style={{ width: "100%" }}
    >
      <MediaProvider>
        {subtitles?.vi && (
          <Track
            src={subtitles.vi}
            kind="subtitles"
            label="Tiếng Việt"
            lang="vi"
            default
          />
        )}
        {subtitles?.en && (
          <Track
            src={subtitles.en}
            kind="subtitles"
            label="English"
            lang="en"
          />
        )}
        {chapters && (
          <Track
            src={chapters}
            kind="chapters"
            label="Phân cảnh"
            lang="vi"
            default
          />
        )}
      </MediaProvider>

      <DefaultVideoLayout
        icons={defaultLayoutIcons}
        showTooltipDelay={0}
        showMenuDelay={0}
        translations={PLAYER_TRANSLATIONS}
      />
    </MediaPlayer>
  );
}
