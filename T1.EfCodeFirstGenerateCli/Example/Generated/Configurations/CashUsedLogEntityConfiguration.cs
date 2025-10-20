using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class CashUsedLogEntityConfiguration : IEntityTypeConfiguration<CashUsedLogEntity>
    {
        public void Configure(EntityTypeBuilder<CashUsedLogEntity> builder)
        {
            builder.ToTable("CashUsedLog");


            builder.Property(x => x.UpdateTime)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.CustomerId)
                .HasColumnType("int(10,0)")
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
