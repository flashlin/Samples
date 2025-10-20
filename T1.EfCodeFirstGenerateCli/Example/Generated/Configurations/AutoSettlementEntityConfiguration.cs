using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class AutoSettlementEntityConfiguration : IEntityTypeConfiguration<AutoSettlementEntity>
    {
        public void Configure(EntityTypeBuilder<AutoSettlementEntity> builder)
        {
            builder.ToTable("AutoSettlement");

            builder.HasKey(x => x.MatchResultId);

            builder.Property(x => x.MatchResultId)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.IsHTReadyForAutoSettle)
                .HasColumnType("bit")
            ;

            builder.Property(x => x.ISFTReadyForAutoSettle)
                .HasColumnType("bit")
            ;

            builder.Property(x => x.HTHomeScore)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.HTAwayScore)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.FinalHomeScore)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.FinalAwayScore)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.FGLG)
                .HasColumnType("nvarchar(2)")
                .HasMaxLength(2)
            ;

            builder.Property(x => x.EventDate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.SettlementStatus)
                .HasColumnType("int(10,0)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.ErrorCode)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.Provider)
                .HasColumnType("nvarchar(20)")
                .HasMaxLength(20)
                .HasDefaultValue(" ")
            ;

            builder.Property(x => x.ProcessingStatus)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
                .HasDefaultValue("Waiting")
            ;

            builder.Property(x => x.SportId)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.SettlementAction)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.CreatedOn)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.LastModifiedOn)
                .HasColumnType("datetime")
            ;

        }
    }
}
