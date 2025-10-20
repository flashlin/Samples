using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class CustomerPromotionEntityConfiguration : IEntityTypeConfiguration<CustomerPromotionEntity>
    {
        public void Configure(EntityTypeBuilder<CustomerPromotionEntity> builder)
        {
            builder.ToTable("CustomerPromotion");

            builder.HasKey(x => x.VoucherId);

            builder.Property(x => x.VoucherId)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.CustId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.PromotionType)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.IsEnabled)
                .HasColumnType("bit")
            ;

            builder.Property(x => x.EffectiveDate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.ExpiryDate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.CashEntitled)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.CashUsed)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.Detail)
                .HasColumnType("varchar(1000)")
                .HasMaxLength(1000)
            ;

            builder.Property(x => x.ModifiedBy)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ModifiedOn)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.CashEntitledInSGD)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.IsRead)
                .HasColumnType("bit")
                .IsRequired()
                .HasDefaultValue(false)
            ;

        }
    }
}
