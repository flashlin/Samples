namespace T1.SqlSchemaExtract;

/// <summary>
/// Represents table schema information retrieved from SQL Server system tables
/// </summary>
public class TableSchemaInfo
{
    /// <summary>
    /// Name of the table
    /// </summary>
    public string TableName { get; set; } = string.Empty;

    /// <summary>
    /// Name of the field/column
    /// </summary>
    public string FieldName { get; set; } = string.Empty;

    /// <summary>
    /// Data type of the field
    /// </summary>
    public string FieldDataType { get; set; } = string.Empty;

    /// <summary>
    /// Maximum length of the field data
    /// </summary>
    public int FieldDataSize { get; set; }

    /// <summary>
    /// Scale of the field data (for decimal types)
    /// </summary>
    public byte FieldDataScale { get; set; }

    /// <summary>
    /// Indicates whether this field is part of the primary key
    /// </summary>
    public bool IsPrimaryKey { get; set; }

    /// <summary>
    /// Indicates whether this field allows null values
    /// </summary>
    public bool IsNullable { get; set; }

    /// <summary>
    /// Indicates whether this field is an identity column
    /// </summary>
    public bool IsIdentity { get; set; }

    /// <summary>
    /// Default value definition for the field
    /// </summary>
    public string DefaultValue { get; set; } = string.Empty;

    /// <summary>
    /// Description of the field from extended properties
    /// </summary>
    public string Description { get; set; } = string.Empty;
}