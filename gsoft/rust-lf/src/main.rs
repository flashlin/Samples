use std::env;
use std::path::Path;
use regex::Regex;
use walkdir::WalkDir;
use colored::*;
use term_size;

fn main() {
    // Get command line arguments
    let args: Vec<String> = env::args().collect();
    
    if args.len() != 2 {
        eprintln!("Usage: {} <regex_pattern>", args[0]);
        eprintln!("Example: {} '.*\\.txt$'", args[0]);
        std::process::exit(1);
    }
    
    let regex_pattern = &args[1];
    
    // Compile regex pattern
    let regex = match Regex::new(regex_pattern) {
        Ok(regex) => regex,
        Err(e) => {
            eprintln!("Invalid regex pattern: {}", e);
            std::process::exit(1);
        }
    };
    
    // Get current directory
    let current_dir = env::current_dir().expect("Failed to get current directory");
    
    // Get terminal width for progress display
    let terminal_width = term_size::dimensions().map(|(w, _)| w).unwrap_or(80);
    
    // Search for files
    search_files(&current_dir, &regex, terminal_width);
    
    // Clear the final progress line after search is complete
    clear_progress_line(terminal_width);
}

fn search_files(current_dir: &Path, regex: &Regex, terminal_width: usize) {
    let walker = WalkDir::new(current_dir).into_iter();
    
    for entry in walker {
        match entry {
            Ok(entry) => {
                let path = entry.path();
                
                // Skip if it's a directory
                if path.is_dir() {
                    // Display progress for directory
                    display_progress(path, terminal_width);
                    continue;
                }
                
                // Check if filename matches regex
                if let Some(file_name) = path.file_name() {
                    if let Some(name_str) = file_name.to_str() {
                        if regex.is_match(name_str) {
                            // Clear the entire progress line first, then display matched file
                            clear_progress_line(terminal_width);
                            display_matched_file(path, name_str, regex);
                        }
                    }
                }
            }
            Err(e) => {
                // Skip files that can't be accessed
                eprintln!("Error accessing path: {}", e);
            }
        }
    }
}

fn display_matched_file(path: &Path, file_name: &str, regex: &Regex) {
    let path_str = path.to_string_lossy();
    
    // Find the position of the filename in the full path
    if let Some(filename_pos) = path_str.rfind(file_name) {
        // Split the path into directory and filename parts
        let directory_part = &path_str[..filename_pos];
        let filename_part = &path_str[filename_pos..];
        
        // Display directory part in white
        print!("{}", directory_part.white());
        
        // Display filename with regex highlighting
        display_highlighted_filename(filename_part, regex);
        println!();
    } else {
        // Fallback: display entire path in white
        println!("{}", path_str.white());
    }
}

fn display_highlighted_filename(filename: &str, regex: &Regex) {
    // Find all matches in the filename
    let mut last_end = 0;
    let mut output = String::new();
    
    for mat in regex.find_iter(filename) {
        // Add non-matching part before this match (in white)
        if mat.start() > last_end {
            output.push_str(&filename[last_end..mat.start()].white().to_string());
        }
        
        // Add matching part (in green)
        output.push_str(&filename[mat.start()..mat.end()].green().to_string());
        
        last_end = mat.end();
    }
    
    // Add remaining non-matching part after last match (in white)
    if last_end < filename.len() {
        output.push_str(&filename[last_end..].white().to_string());
    }
    
    // If no matches found, display entire filename in white
    if last_end == 0 {
        print!("{}", filename.white());
    } else {
        print!("{}", output);
    }
}

fn clear_progress_line(terminal_width: usize) {
    // Move cursor to beginning of line
    print!("\r");
    // Print spaces to clear the entire line
    print!("{}", " ".repeat(terminal_width));
    // Move cursor back to beginning of line
    print!("\r");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
}

fn display_progress(path: &Path, terminal_width: usize) {
    let path_str = path.to_string_lossy();
    
    // Calculate display width (account for progress indicator)
    let progress_width = 20; // Space for "Searching: " and "\r"
    let available_width = terminal_width.saturating_sub(progress_width);
    
    let display_path = if path_str.len() > available_width {
        // Truncate and show only the end part
        let truncated = path_str.len() - available_width;
        format!("...{}", &path_str[truncated..])
    } else {
        path_str.to_string()
    };
    
    // Print progress on the same line
    print!("\rSearching: {}", display_path);
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
}
