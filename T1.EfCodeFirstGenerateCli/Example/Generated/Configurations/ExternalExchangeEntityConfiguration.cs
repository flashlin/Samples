using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class ExternalExchangeEntityConfiguration : IEntityTypeConfiguration<ExternalExchangeEntity>
    {
        public void Configure(EntityTypeBuilder<ExternalExchangeEntity> builder)
        {
            builder.ToTable("ExternalExchange");

            builder.HasKey(x => new { x.EffectiveDate, x.Currency });

            builder.Property(x => x.EffectiveDate)
                .HasColumnType("smalldatetime")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.SystemRate)
                .HasColumnType("")
                .IsRequired()
            ;

            builder.Property(x => x.ExternalRate)
                .HasColumnType("")
                .IsRequired()
            ;

            builder.Property(x => x.Currency)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.CurrencyStr)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ModifiedDate)
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
