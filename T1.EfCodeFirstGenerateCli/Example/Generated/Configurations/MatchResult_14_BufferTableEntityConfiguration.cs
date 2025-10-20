using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class MatchResult_14_BufferTableEntityConfiguration : IEntityTypeConfiguration<MatchResult_14_BufferTableEntity>
    {
        public void Configure(EntityTypeBuilder<MatchResult_14_BufferTableEntity> builder)
        {
            builder.ToTable("MatchResult_14_BufferTable");


            builder.Property(x => x.MatchResultId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.MatchId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.LeagueId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.HomeId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.AwayId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.BetTypeGroupId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.EventDate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.EventStatus)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.LiveHomeScore)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.LiveAwayScore)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.FinalHomeScore)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.FinalAwayScore)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.Creator)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.MatchCode)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.KickOffTime)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.ShowTime)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.HTHomeScore)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.HTAwayScore)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.Ruben)
                .HasColumnType("tinyint(3,0)")
            ;

            builder.Property(x => x.Multiple)
                .HasColumnType("bit")
            ;

            builder.Property(x => x.tstamp)
                .HasColumnType("binary(8)")
                .HasMaxLength(8)
            ;

            builder.Property(x => x.SportId)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.Result)
                .HasColumnType("nvarchar(200)")
                .HasMaxLength(200)
            ;

            builder.Property(x => x.Color)
                .HasColumnType("nvarchar(10)")
                .HasMaxLength(10)
            ;

            builder.Property(x => x.Remark)
                .HasColumnType("nvarchar(1000)")
                .HasMaxLength(1000)
            ;

            builder.Property(x => x.OtherStatus)
                .HasColumnType("tinyint(3,0)")
            ;

            builder.Property(x => x.OtherStatus2)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.AbsKickoffTime)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.EventType)
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
            ;

            builder.Property(x => x.EventTypeID)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.LiveHdpTrader)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.LiveOuTrader)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.HdpTrader)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.OuTrader)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.ft1x2Trader)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.fh1x2Trader)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.ShowTimeDisplayType)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.LiveFhHdpTrader)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.LiveFhOuTrader)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.CreateOn)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.HomeJersey)
                .HasColumnType("nvarchar(1)")
                .HasMaxLength(1)
            ;

            builder.Property(x => x.AwayJersey)
                .HasColumnType("nvarchar(1)")
                .HasMaxLength(1)
            ;

            builder.Property(x => x.ParentId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.LastModifiedOn)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.Channel)
                .HasColumnType("nvarchar(512)")
                .HasMaxLength(512)
            ;

            builder.Property(x => x.ScoreVerified)
                .HasColumnType("bit")
            ;

            builder.Property(x => x.LiveCsTrader)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.LiveFhCsTrader)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.OwnTstamp)
                .HasColumnType("timestamp")
                .IsRequired()
            ;

        }
    }
}
