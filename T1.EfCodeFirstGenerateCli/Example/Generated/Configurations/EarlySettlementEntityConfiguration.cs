using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class EarlySettlementEntityConfiguration : IEntityTypeConfiguration<EarlySettlementEntity>
    {
        public void Configure(EntityTypeBuilder<EarlySettlementEntity> builder)
        {
            builder.ToTable("EarlySettlement");

            builder.HasKey(x => new { x.MatchResultId, x.SettlementTime });

            builder.Property(x => x.MatchResultId)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.HomeScore)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.AwayScore)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.SettlementTime)
                .HasColumnType("datetime")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.Period)
                .HasColumnType("int(10,0)")
            ;

        }
    }
}
