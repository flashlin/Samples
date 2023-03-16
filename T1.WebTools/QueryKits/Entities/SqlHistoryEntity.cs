using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace QueryKits.Entities;

[Table("SqlHistory")]
public class SqlHistoryEntity
{
    [Key]
    [DatabaseGenerated(DatabaseGeneratedOption.Identity)]
    public int Id { get; set; }
    [Column(TypeName = "NVARCHAR(2000)")] public string SqlCode { get; set; } = string.Empty;
    public DateTime CreatedOn{get; set;}
}