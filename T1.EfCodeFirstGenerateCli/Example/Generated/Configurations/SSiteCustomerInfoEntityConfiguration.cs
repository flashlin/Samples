using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class SSiteCustomerInfoEntityConfiguration : IEntityTypeConfiguration<SSiteCustomerInfoEntity>
    {
        public void Configure(EntityTypeBuilder<SSiteCustomerInfoEntity> builder)
        {
            builder.ToTable("SSiteCustomerInfo");

            builder.HasKey(x => x.CustomerId);

            builder.Property(x => x.CustomerId)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.FirstName)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.LastName)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.Phone)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.MobilePhone)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

        }
    }
}
