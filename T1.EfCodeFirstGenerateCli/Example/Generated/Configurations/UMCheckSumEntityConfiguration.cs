using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class UMCheckSumEntityConfiguration : IEntityTypeConfiguration<UMCheckSumEntity>
    {
        public void Configure(EntityTypeBuilder<UMCheckSumEntity> builder)
        {
            builder.ToTable("UMCheckSum");


            builder.Property(x => x.Winlostdate)
                .HasColumnType("smalldatetime")
                .IsRequired()
            ;

            builder.Property(x => x.CheckSumFloat)
                .HasColumnType("")
            ;

            builder.Property(x => x.CheckType)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.CheckSumDecimal)
                .HasColumnType("decimal(22,6)")
                .IsRequired()
            ;

            builder.Property(x => x.CheckTotalSumDecimal)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.CreatedDate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.CheckCasinoTotalSumDecimal)
                .HasColumnType("decimal(19,6)")
            ;

        }
    }
}
