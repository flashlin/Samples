using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class ExchangeCodeLookUpEntityConfiguration : IEntityTypeConfiguration<ExchangeCodeLookUpEntity>
    {
        public void Configure(EntityTypeBuilder<ExchangeCodeLookUpEntity> builder)
        {
            builder.ToTable("ExchangeCodeLookUp");


            builder.Property(x => x.ExchangeID)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.CurrencyCode)
                .HasColumnType("nchar(10)")
                .HasMaxLength(10)
            ;

        }
    }
}
