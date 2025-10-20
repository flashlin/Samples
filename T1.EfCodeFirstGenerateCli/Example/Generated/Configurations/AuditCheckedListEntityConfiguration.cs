using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class AuditCheckedListEntityConfiguration : IEntityTypeConfiguration<AuditCheckedListEntity>
    {
        public void Configure(EntityTypeBuilder<AuditCheckedListEntity> builder)
        {
            builder.ToTable("AuditCheckedList");

            builder.HasKey(x => x.rid);

            builder.Property(x => x.rid)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.custid)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.auditor)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.cashbalance)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.totalbalance)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.auditDate)
                .HasColumnType("smalldatetime")
                .IsRequired()
            ;

            builder.Property(x => x.remark)
                .HasColumnType("nvarchar(100)")
                .HasMaxLength(100)
            ;

            builder.Property(x => x.createdDate)
                .HasColumnType("smalldatetime")
                .IsRequired()
            ;

            builder.Property(x => x.unCheckAudit)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

        }
    }
}
