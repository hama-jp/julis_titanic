"""
Submission File Helper

このヘルパーは、提出ファイルを適切なディレクトリに保存することを保証します。
ルートフォルダにCSVファイルが散乱することを防ぎます。

使い方:
    from scripts.utils.submission_helper import save_submission

    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': predictions
    })

    # ✅ 正しい：submissions/に自動保存
    save_submission(submission, 'my_model_predictions.csv')

    # ❌ 間違い：ルートに保存（非推奨）
    # submission.to_csv('my_model_predictions.csv', index=False)
"""

import pandas as pd
import os
from datetime import datetime
from pathlib import Path
import warnings


def get_submissions_dir() -> Path:
    """
    提出ファイルのディレクトリパスを取得

    Returns
    -------
    submissions_dir : Path
        submissions/ ディレクトリのパス
    """
    # プロジェクトルートを取得
    project_root = Path(__file__).resolve().parents[2]
    submissions_dir = project_root / 'submissions'

    # ディレクトリが存在しない場合は作成
    submissions_dir.mkdir(exist_ok=True)

    return submissions_dir


def save_submission(
    df: pd.DataFrame,
    filename: str,
    add_timestamp: bool = False,
    overwrite: bool = False
) -> Path:
    """
    提出ファイルをsubmissions/ディレクトリに保存

    Parameters
    ----------
    df : pd.DataFrame
        提出データ（PassengerId, Survivedカラムが必要）
    filename : str
        ファイル名（拡張子 .csv を含む）
    add_timestamp : bool, default=False
        ファイル名にタイムスタンプを追加するか
    overwrite : bool, default=False
        既存ファイルの上書きを許可するか

    Returns
    -------
    filepath : Path
        保存されたファイルのパス

    Raises
    ------
    ValueError
        必要なカラムが不足している場合
    FileExistsError
        overwrite=False で既存ファイルが存在する場合

    Examples
    --------
    >>> submission = pd.DataFrame({
    ...     'PassengerId': [892, 893, 894],
    ...     'Survived': [0, 1, 0]
    ... })
    >>> filepath = save_submission(submission, 'my_predictions.csv')
    >>> print(filepath)
    submissions/my_predictions.csv

    >>> # タイムスタンプ付き
    >>> filepath = save_submission(submission, 'model.csv', add_timestamp=True)
    >>> print(filepath)
    submissions/20251024_1234_model.csv
    """
    # カラムの検証
    required_columns = ['PassengerId', 'Survived']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Submission file must have 'PassengerId' and 'Survived' columns."
        )

    # 行数の検証（Titanicテストデータは418行）
    expected_rows = 418
    if len(df) != expected_rows:
        warnings.warn(
            f"Expected {expected_rows} rows but got {len(df)} rows. "
            f"Titanic test set should have 418 passengers.",
            UserWarning
        )

    # ファイル名の処理
    if not filename.endswith('.csv'):
        filename = f"{filename}.csv"

    # タイムスタンプ追加
    if add_timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        base_name = filename.replace('.csv', '')
        filename = f"{timestamp}_{base_name}.csv"

    # 保存先ディレクトリ
    submissions_dir = get_submissions_dir()
    filepath = submissions_dir / filename

    # 上書きチェック
    if filepath.exists() and not overwrite:
        raise FileExistsError(
            f"File already exists: {filepath}\n"
            f"Set overwrite=True to overwrite, or use a different filename."
        )

    # 保存
    df[required_columns].to_csv(filepath, index=False)

    print(f"✅ Submission saved: {filepath}")
    print(f"   Rows: {len(df)}")
    print(f"   Survival rate: {df['Survived'].mean():.3f}")

    return filepath


def archive_submission(filename: str) -> Path:
    """
    提出ファイルをアーカイブディレクトリに移動

    Parameters
    ----------
    filename : str
        アーカイブするファイル名

    Returns
    -------
    new_path : Path
        アーカイブされたファイルのパス
    """
    submissions_dir = get_submissions_dir()
    archive_dir = submissions_dir / 'archive'
    archive_dir.mkdir(exist_ok=True)

    source = submissions_dir / filename
    destination = archive_dir / filename

    if not source.exists():
        raise FileNotFoundError(f"File not found: {source}")

    if destination.exists():
        warnings.warn(f"Overwriting existing archive file: {destination}")

    source.rename(destination)
    print(f"📦 Archived: {filename} → archive/")

    return destination


def list_submissions(pattern: str = "*.csv", sort_by: str = "time") -> list:
    """
    提出ファイルの一覧を取得

    Parameters
    ----------
    pattern : str, default="*.csv"
        ファイルパターン（glob形式）
    sort_by : str, default="time"
        ソート方法（"time" または "name"）

    Returns
    -------
    files : list
        ファイルパスのリスト
    """
    submissions_dir = get_submissions_dir()
    files = list(submissions_dir.glob(pattern))

    # ディレクトリを除外
    files = [f for f in files if f.is_file()]

    if sort_by == "time":
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    elif sort_by == "name":
        files.sort()

    return files


def warn_if_root_csv():
    """
    ルートディレクトリにCSVファイルがある場合に警告

    この関数をスクリプトの最後に呼び出すことで、
    誤ってルートに保存されたファイルを検出できます。
    """
    project_root = Path(__file__).resolve().parents[2]
    root_csvs = list(project_root.glob("*.csv"))

    # data/raw/ の train.csv, test.csv は除外
    root_csvs = [f for f in root_csvs if f.name not in ['train.csv', 'test.csv']]

    if root_csvs:
        warnings.warn(
            f"\n{'='*80}\n"
            f"⚠️  WARNING: CSV files found in project root!\n"
            f"{'='*80}\n"
            f"The following files should be in submissions/ directory:\n"
            + "\n".join(f"  - {f.name}" for f in root_csvs) +
            f"\n\nPlease use save_submission() to save to submissions/ directory.\n"
            f"See: scripts/utils/submission_helper.py\n"
            f"{'='*80}\n",
            UserWarning,
            stacklevel=2
        )


if __name__ == "__main__":
    """
    使用例とテスト
    """
    print("=" * 80)
    print("SUBMISSION HELPER - DEMO")
    print("=" * 80)

    # テストデータ作成
    test_submission = pd.DataFrame({
        'PassengerId': range(892, 892 + 418),
        'Survived': [0, 1] * 209  # 交互に0と1
    })

    print("\n✅ Example 1: Basic save")
    try:
        filepath = save_submission(
            test_submission,
            'test_submission.csv',
            overwrite=True
        )
        print(f"Saved to: {filepath}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n✅ Example 2: Save with timestamp")
    try:
        filepath = save_submission(
            test_submission,
            'timestamped_model.csv',
            add_timestamp=True,
            overwrite=True
        )
        print(f"Saved to: {filepath}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n✅ Example 3: List submissions")
    files = list_submissions()
    print(f"Found {len(files)} submission files:")
    for f in files[:5]:  # 最新5件のみ表示
        print(f"  - {f.name}")

    print("\n✅ Example 4: Check for root CSV files")
    warn_if_root_csv()

    print("\n" + "=" * 80)
    print("Demo completed!")
    print("=" * 80)
