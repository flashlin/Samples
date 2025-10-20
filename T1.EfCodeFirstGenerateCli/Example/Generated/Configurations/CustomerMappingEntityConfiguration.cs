using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class CustomerMappingEntityConfiguration : IEntityTypeConfiguration<CustomerMappingEntity>
    {
        public void Configure(EntityTypeBuilder<CustomerMappingEntity> builder)
        {
            builder.ToTable("CustomerMapping");

            builder.HasKey(x => new { x.CustomerId, x.ParentId });

            builder.Property(x => x.CustomerId)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.ParentId)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

        }
    }
}
