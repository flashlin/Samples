declare module 'sqlite-wasm-esm' {
    interface Sqlite3 {
        opfs: {
            OpfsDb: {
                new(dbName: string, mode: string): OpfsDb;
            };
        };
    }

    interface OpfsDb {
        exec(query: string): any;
    }

    const sqlite3InitModule: () => Promise<Sqlite3>;
    export default sqlite3InitModule;
}
