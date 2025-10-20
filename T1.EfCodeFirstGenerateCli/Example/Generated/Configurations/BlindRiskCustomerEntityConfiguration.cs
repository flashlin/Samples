using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class BlindRiskCustomerEntityConfiguration : IEntityTypeConfiguration<BlindRiskCustomerEntity>
    {
        public void Configure(EntityTypeBuilder<BlindRiskCustomerEntity> builder)
        {
            builder.ToTable("BlindRiskCustomer");


            builder.Property(x => x.CustId)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.IsEnabled)
                .HasColumnType("bit")
            ;

            builder.Property(x => x.IsHighRiskPlayer)
                .HasColumnType("bit")
            ;

            builder.Property(x => x.Score)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.BlindRiskRate)
                .HasColumnType("decimal(3,2)")
            ;

            builder.Property(x => x.ModifiedOn)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.Remark)
                .HasColumnType("varchar(500)")
                .HasMaxLength(500)
            ;

        }
    }
}
