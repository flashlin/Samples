mod sqlitex;

fn main() {
    let db = sqlitex::Sqlitex::SqliteDb::new(& str"test.db");
    //let db = sqlitex::Sqlitex::SqliteDb::new(String::from("test.db"));
    println!("Hello, world!");
}
