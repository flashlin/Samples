using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class BlindRiskCustomerClusterEntityConfiguration : IEntityTypeConfiguration<BlindRiskCustomerClusterEntity>
    {
        public void Configure(EntityTypeBuilder<BlindRiskCustomerClusterEntity> builder)
        {
            builder.ToTable("BlindRiskCustomerCluster");

            builder.HasKey(x => x.CustId);

            builder.Property(x => x.CustId)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.AccumulatedTurnover)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.StakeRatio)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.HighBrApplied)
                .HasColumnType("bit")
                .IsRequired()
                .HasDefaultValue(false)
            ;

        }
    }
}
