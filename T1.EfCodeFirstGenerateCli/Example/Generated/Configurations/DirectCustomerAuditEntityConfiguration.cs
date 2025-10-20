using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class DirectCustomerAuditEntityConfiguration : IEntityTypeConfiguration<DirectCustomerAuditEntity>
    {
        public void Configure(EntityTypeBuilder<DirectCustomerAuditEntity> builder)
        {
            builder.ToTable("DirectCustomerAudit");

            builder.HasKey(x => new { x.DirectCustID, x.AuditDate });

            builder.Property(x => x.DirectCustID)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.Auditor)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.CashBalance)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
            ;

            builder.Property(x => x.TotalBalance)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
            ;

            builder.Property(x => x.AuditDate)
                .HasColumnType("smalldatetime")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.Remark)
                .HasColumnType("nvarchar(100)")
                .HasMaxLength(100)
            ;

            builder.Property(x => x.CreatedOn)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.CreatedBy)
                .HasColumnType("varchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ModifiedBy)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ModifiedOn)
                .HasColumnType("datetime")
            ;

        }
    }
}
