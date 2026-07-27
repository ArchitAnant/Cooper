from pathlib import Path
import subprocess


def convert_mp3_to_wav(input_path: Path, output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	subprocess.run(
		[
			"ffmpeg",
			"-y",
			"-i",
			str(input_path),
			str(output_path),
		],
		check=True,
		stdout=subprocess.DEVNULL,
		stderr=subprocess.DEVNULL,
	)


def main() -> None:
	base_dir = Path("testing_audio")
	if not base_dir.exists():
		return

	for class_dir in base_dir.iterdir():
		if not class_dir.is_dir():
			continue

		mp3_files = sorted(class_dir.glob("*.mp3"))
		for index, mp3_file in enumerate(mp3_files, start=1):
			wav_file = class_dir / f"{index}.wav"
			convert_mp3_to_wav(mp3_file, wav_file)


if __name__ == "__main__":
	main()
