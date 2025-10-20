using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class CashUsedLogSumEntityConfiguration : IEntityTypeConfiguration<CashUsedLogSumEntity>
    {
        public void Configure(EntityTypeBuilder<CashUsedLogSumEntity> builder)
        {
            builder.ToTable("CashUsedLogSum");


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

            builder.Property(x => x.Amount)
                .HasColumnType("decimal(19,6)")
                .HasDefaultValue(0)
            ;

        }
    }
}
