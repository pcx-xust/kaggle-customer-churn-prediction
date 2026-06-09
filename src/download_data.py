from pathlib import Path
import shutil
import kagglehub


COMPETITION_NAME = "playground-series-s6e3"


def main():
    project_dir = Path(__file__).resolve().parents[1]
    raw_dir = project_dir / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    download_dir = Path(kagglehub.competition_download(COMPETITION_NAME))

    expected_files = [
        "train.csv",
        "test.csv",
        "sample_submission.csv",
    ]

    for file_name in expected_files:
        src_path = download_dir / file_name
        dst_path = raw_dir / file_name

        if not src_path.exists():
            print(f"[Missing] {src_path}")
            continue

        shutil.copy2(src_path, dst_path)
        print(f"[Copied] {file_name} -> {dst_path}")

    print("Data download finished.")


if __name__ == "__main__":
    main()
