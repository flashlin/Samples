using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class RiskControlBalanceEntityConfiguration : IEntityTypeConfiguration<RiskControlBalanceEntity>
    {
        public void Configure(EntityTypeBuilder<RiskControlBalanceEntity> builder)
        {
            builder.ToTable("RiskControlBalance");

            builder.HasKey(x => x.CustomerId);

            builder.Property(x => x.CustomerId)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.WinLose)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.LastResetTime)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.ModifiedOn)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.ParentId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

        }
    }
}
