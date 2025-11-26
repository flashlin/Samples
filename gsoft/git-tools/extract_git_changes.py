#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def validate_git_repository(git_folder_path):
    git_folder = Path(git_folder_path)
    if not git_folder.exists():
        raise ValueError(f"Path does not exist: {git_folder_path}")
    
    git_dir = git_folder / '.git'
    if not git_dir.exists():
        raise ValueError(f"Not a valid git repository: {git_folder_path}")


def validate_commit(git_folder_path, commit_hash):
    try:
        subprocess.run(
            ['git', 'rev-parse', '--verify', commit_hash],
            cwd=git_folder_path,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except subprocess.CalledProcessError:
        raise ValueError(f"Invalid commit hash: {commit_hash}")


def get_changed_files(git_folder_path, start_commit, end_commit):
    try:
        result = subprocess.run(
            ['git', 'diff', '--name-status', '--diff-filter=ADM', start_commit, end_commit],
            cwd=git_folder_path,
            check=True,
            capture_output=True,
            text=True
        )
        return result.stdout.strip().split('\n') if result.stdout.strip() else []
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to get changed files: {e.stderr}")


def parse_file_changes(changed_files_output):
    added_modified = []
    deleted = []
    
    for line in changed_files_output:
        if not line.strip():
            continue
        
        status = line[0]
        file_path = line[1:].strip()
        
        if status == 'D':
            deleted.append(file_path)
        elif status in ['A', 'M']:
            added_modified.append(file_path)
    
    return added_modified, deleted


def extract_file_content(git_folder_path, commit_hash, file_path):
    try:
        result = subprocess.run(
            ['git', 'show', f'{commit_hash}:{file_path}'],
            cwd=git_folder_path,
            check=True,
            capture_output=True,
            text=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to extract file {file_path}: {e.stderr}")


def create_output_directory_structure(output_path, file_path):
    full_path = Path(output_path) / file_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    return full_path


def save_file_content(output_path, file_path, content):
    full_path = create_output_directory_structure(output_path, file_path)
    full_path.write_text(content, encoding='utf-8')
    print(f"Extracted: {file_path}")


def save_deleted_files_list(output_path, deleted_files):
    deleted_log_path = Path(output_path) / 'deleted.log'
    with open(deleted_log_path, 'w', encoding='utf-8') as f:
        for file_path in deleted_files:
            f.write(f"{file_path}\n")
    print(f"Deleted files list saved to: {deleted_log_path}")


def extract_changes(git_folder_path, start_commit, end_commit, output_path):
    validate_git_repository(git_folder_path)
    validate_commit(git_folder_path, start_commit)
    validate_commit(git_folder_path, end_commit)
    
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    changed_files_output = get_changed_files(git_folder_path, start_commit, end_commit)
    added_modified, deleted = parse_file_changes(changed_files_output)
    
    for file_path in added_modified:
        try:
            content = extract_file_content(git_folder_path, end_commit, file_path)
            save_file_content(output_path, file_path, content)
        except RuntimeError as e:
            print(f"Warning: {e}", file=sys.stderr)
    
    if deleted:
        save_deleted_files_list(output_path, deleted)
    
    print(f"\nExtraction completed:")
    print(f"  Added/Modified files: {len(added_modified)}")
    print(f"  Deleted files: {len(deleted)}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract changed files from git commit range'
    )
    parser.add_argument(
        'git_folder_path',
        help='Path to git repository folder'
    )
    parser.add_argument(
        'start_commit',
        help='Start commit hashid'
    )
    parser.add_argument(
        'end_commit',
        help='End commit hashid'
    )
    parser.add_argument(
        'output_path',
        help='Output directory path'
    )
    
    args = parser.parse_args()
    
    try:
        extract_changes(
            args.git_folder_path,
            args.start_commit,
            args.end_commit,
            args.output_path
        )
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

