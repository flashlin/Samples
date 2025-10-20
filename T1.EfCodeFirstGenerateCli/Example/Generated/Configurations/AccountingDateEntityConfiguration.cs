using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class AccountingDateEntityConfiguration : IEntityTypeConfiguration<AccountingDateEntity>
    {
        public void Configure(EntityTypeBuilder<AccountingDateEntity> builder)
        {
            builder.ToTable("AccountingDate");

            builder.HasKey(x => new { x.ActionType, x.ProductType });

            builder.Property(x => x.ActionType)
                .HasColumnType("nvarchar(50)")
                .ValueGeneratedOnAdd()
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ProductType)
                .HasColumnType("tinyint(3,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.LastActionDate)
                .HasColumnType("smalldatetime")
            ;

            builder.Property(x => x.ModifiedOn)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.ModifiedBy)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

        }
    }
}
