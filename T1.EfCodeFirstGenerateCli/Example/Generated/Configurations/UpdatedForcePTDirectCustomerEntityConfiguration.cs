using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class UpdatedForcePTDirectCustomerEntityConfiguration : IEntityTypeConfiguration<UpdatedForcePTDirectCustomerEntity>
    {
        public void Configure(EntityTypeBuilder<UpdatedForcePTDirectCustomerEntity> builder)
        {
            builder.ToTable("UpdatedForcePTDirectCustomer");

            builder.HasKey(x => x.CustID);

            builder.Property(x => x.CustID)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.AccountType)
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
            ;

        }
    }
}
