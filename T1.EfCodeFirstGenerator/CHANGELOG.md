# Changelog

All notable changes to T1.EfCodeFirstGenerateCli will be documented in this file.

## [1.0.0] - 2025-10-20

### Added
- Initial release of T1.EfCodeFirstGenerateCli
- CLI tool for database schema extraction and code generation
- MSBuild Task integration for automatic code generation during build
- Support for SQL Server and MySQL databases
- Automatic generation of:
  - DbContext with DbSet properties
  - Entity classes with proper C# types
  - EntityConfiguration classes with Fluent API
- Schema caching system to avoid repeated database connections
- NuGet package support for easy distribution
- Comprehensive documentation (README, USAGE_GUIDE)
- Example project demonstrating usage

### Features
- Automatic SQL type to C# type mapping
- Support for primary keys, nullable fields, default values
- Configurable namespace generation
- `required` modifier for non-nullable reference types
- Extensible type converter system
- Multiple database support in single project
- Git-friendly workflow with schema file caching

### Supported Databases
- SQL Server ✅
- MySQL / MariaDB ✅
- PostgreSQL (planned)
- Oracle (planned)

### Known Issues
- Microsoft.Build.Utilities.Core has a known vulnerability (acceptable for development tools)

### Dependencies
- .NET 8.0
- System.Data.SqlClient 4.8.6
- MySql.Data 8.3.0
- Newtonsoft.Json 13.0.3
- Microsoft.Build.Utilities.Core 17.11.4

## [Unreleased]

### Planned Features
- PostgreSQL support
- Oracle support
- Support for foreign key relationships
- Support for indexes and constraints
- Custom template system
- Visual Studio extension
- dotnet tool global installation

