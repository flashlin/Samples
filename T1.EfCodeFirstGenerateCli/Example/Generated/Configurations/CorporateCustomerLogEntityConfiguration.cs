using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class CorporateCustomerLogEntityConfiguration : IEntityTypeConfiguration<CorporateCustomerLogEntity>
    {
        public void Configure(EntityTypeBuilder<CorporateCustomerLogEntity> builder)
        {
            builder.ToTable("CorporateCustomerLog");


            builder.Property(x => x.CustomerId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.CorporateGroupId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.StartDate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.EndDate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.ModifiedBy)
                .HasColumnType("varchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ModifiedOn)
                .HasColumnType("datetime")
                .IsRequired()
            ;

        }
    }
}
