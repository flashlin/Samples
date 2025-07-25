use regex::Regex;
use std::env;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use walkdir::WalkDir;
use colored::*;

fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() != 3 {
        eprintln!("Usage: {} <content_regex> <filename_regex>", args[0]);
        eprintln!("Example: {} \"error\" \".*\\.log$\"", args[0]);
        eprintln!("  content_regex: Pattern to search within file content");
        eprintln!("  filename_regex: Pattern to match filenames");
        std::process::exit(1);
    }
    
    let content_pattern = &args[1];
    let filename_pattern = &args[2];
    
    // Compile the regex patterns
    let content_regex = match Regex::new(content_pattern) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Content regex error: {}", e);
            std::process::exit(1);
        }
    };
    
    let filename_regex = match Regex::new(filename_pattern) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Filename regex error: {}", e);
            std::process::exit(1);
        }
    };
    
    println!("Searching for content pattern '{}' in files matching '{}'", content_pattern, filename_pattern);
    println!("Search directory: {}", env::current_dir().unwrap().display());
    println!();
    
    let mut total_matches = 0;
    let mut files_processed = 0;
    
    // Walk through current directory and subdirectories
    for entry in WalkDir::new(".").into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();
        
        // Show progress for directories
        if path.is_dir() {
            print!("\rSearching: {}", path.display());
            io::stdout().flush().unwrap();
        }
        
        // Only process files, skip directories
        if path.is_file() {
            files_processed += 1;
            
            if let Some(file_name) = path.file_name() {
                if let Some(file_name_str) = file_name.to_str() {
                    // Check if filename matches the regex pattern
                    if filename_regex.is_match(file_name_str) {
                        // Show current file being processed
                        print!("\rProcessing: {}", path.display());
                        io::stdout().flush().unwrap();
                        
                        // Search content within the file
                        match search_file_content(&path, &content_regex, content_pattern) {
                            Ok(matches) => {
                                if matches > 0 {
                                    total_matches += matches;
                                }
                            }
                            Err(e) => {
                                eprintln!("\nError reading file {}: {}", path.display(), e);
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Clear the progress line
    print!("\r{}\r", " ".repeat(80));
    
    println!();
    if total_matches == 0 {
        println!("No content matches found.");
    } else {
        println!("Total matches found: {} (processed {} files)", total_matches, files_processed);
    }
}

fn search_file_content(path: &std::path::Path, regex: &Regex, _pattern: &str) -> io::Result<usize> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut matches = 0;
    
    for (line_number, line) in reader.lines().enumerate() {
        let line = line?;
        
        if regex.is_match(&line) {
            matches += 1;
            
            // Print filename, line number, and highlighted content
            print!("{} {} ", 
                path.display().to_string().blue().bold(),
                format!("{}", line_number + 1).yellow().bold()
            );
            
            // Highlight matching parts in green, rest in white
            let mut last_end = 0;
            for mat in regex.find_iter(&line) {
                // Print text before match in white
                print!("{}", &line[last_end..mat.start()].white());
                // Print match in green
                print!("{}", &line[mat.start()..mat.end()].green().bold());
                last_end = mat.end();
            }
            // Print remaining text in white
            println!("{}", &line[last_end..].white());
        }
    }
    
    Ok(matches)
}