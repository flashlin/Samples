using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class MaxPTConfigEntityConfiguration : IEntityTypeConfiguration<MaxPTConfigEntity>
    {
        public void Configure(EntityTypeBuilder<MaxPTConfigEntity> builder)
        {
            builder.ToTable("MaxPTConfig");

            builder.HasKey(x => x.Id);

            builder.Property(x => x.Id)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.CurrencyId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.Currency)
                .HasColumnType("varchar(10)")
                .IsRequired()
                .HasMaxLength(10)
            ;

            builder.Property(x => x.ProductType)
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
            ;

            builder.Property(x => x.AccountType)
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
            ;

            builder.Property(x => x.MaxPT)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.ModifiedDate)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.LeoEnumValue)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.SmaId)
                .HasColumnType("int(10,0)")
            ;

        }
    }
}
