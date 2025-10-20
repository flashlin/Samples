using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class AccountingCheckSumEntityConfiguration : IEntityTypeConfiguration<AccountingCheckSumEntity>
    {
        public void Configure(EntityTypeBuilder<AccountingCheckSumEntity> builder)
        {
            builder.ToTable("AccountingCheckSum");


            builder.Property(x => x.StatementDate)
                .HasColumnType("smalldatetime")
                .IsRequired()
            ;

            builder.Property(x => x.StatementType)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.ProductType)
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
            ;

            builder.Property(x => x.DailyTotalRaw)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.DailyTotalSma)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.DailyTotalCash)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.ModifiedBy)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ModifiedOn)
                .HasColumnType("datetime")
                .IsRequired()
            ;

        }
    }
}
