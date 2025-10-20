using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class FantasyDailyWinLostEntityConfiguration : IEntityTypeConfiguration<FantasyDailyWinLostEntity>
    {
        public void Configure(EntityTypeBuilder<FantasyDailyWinLostEntity> builder)
        {
            builder.ToTable("FantasyDailyWinLost");

            builder.HasKey(x => new { x.CustId, x.MatchId });

            builder.Property(x => x.CustId)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.MatchId)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.WinLost)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.NumberOfWinBets)
                .HasColumnType("int(10,0)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

        }
    }
}
