mod sqlitex;

fn main() {
    let db = sqlitex::Sqlitex::SqliteDb { 
        name: String::from("test.db"),
    };
    println!("Hello, world!");
}
