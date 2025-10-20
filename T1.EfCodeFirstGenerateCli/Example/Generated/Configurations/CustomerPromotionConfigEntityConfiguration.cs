using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class CustomerPromotionConfigEntityConfiguration : IEntityTypeConfiguration<CustomerPromotionConfigEntity>
    {
        public void Configure(EntityTypeBuilder<CustomerPromotionConfigEntity> builder)
        {
            builder.ToTable("CustomerPromotionConfig");


            builder.Property(x => x.ID)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.CustID)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.PromotionType)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.Target)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
            ;

            builder.Property(x => x.CreatedOn)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.CreatedBy)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ModifiedOn)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.ModifiedBy)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

        }
    }
}
