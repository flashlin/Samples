using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class FantasyRankingEntityConfiguration : IEntityTypeConfiguration<FantasyRankingEntity>
    {
        public void Configure(EntityTypeBuilder<FantasyRankingEntity> builder)
        {
            builder.ToTable("FantasyRanking");

            builder.HasKey(x => x.CustId);

            builder.Property(x => x.CustId)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.FirstName)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.NetWinLost)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
            ;

            builder.Property(x => x.Ranking)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.PreviousRanking)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.LastModifiedOn)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.Email)
                .HasColumnType("nvarchar(300)")
                .HasMaxLength(300)
            ;

            builder.Property(x => x.NumberOfWinBets)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

        }
    }
}
