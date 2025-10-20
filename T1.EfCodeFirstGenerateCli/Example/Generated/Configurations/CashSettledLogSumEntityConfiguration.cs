using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class CashSettledLogSumEntityConfiguration : IEntityTypeConfiguration<CashSettledLogSumEntity>
    {
        public void Configure(EntityTypeBuilder<CashSettledLogSumEntity> builder)
        {
            builder.ToTable("CashSettledLogSum");


            builder.Property(x => x.CustomerId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.UpdateTime)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.ProductType)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.SettledAmount)
                .HasColumnType("decimal(19,6)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.ReturnAmount)
                .HasColumnType("decimal(19,6)")
                .HasDefaultValue(0)
            ;

        }
    }
}
