using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class IomCustomersAllowedForSBOProductsEntityConfiguration : IEntityTypeConfiguration<IomCustomersAllowedForSBOProductsEntity>
    {
        public void Configure(EntityTypeBuilder<IomCustomersAllowedForSBOProductsEntity> builder)
        {
            builder.ToTable("IomCustomersAllowedForSBOProducts");

            builder.HasKey(x => new { x.CustomerId, x.CreatedOn });

            builder.Property(x => x.CustomerId)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.CreatedBy)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.CreatedOn)
                .HasColumnType("datetime")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.EndedOn)
                .HasColumnType("datetime")
            ;

        }
    }
}
