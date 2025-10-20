using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class AutoSettlementRecordEntityConfiguration : IEntityTypeConfiguration<AutoSettlementRecordEntity>
    {
        public void Configure(EntityTypeBuilder<AutoSettlementRecordEntity> builder)
        {
            builder.ToTable("AutoSettlementRecord");

            builder.HasKey(x => x.Id);

            builder.Property(x => x.Id)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.MatchResultId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.Result)
                .HasColumnType("nvarchar(1000)")
                .HasMaxLength(1000)
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

            builder.Property(x => x.EventDate)
                .HasColumnType("datetime")
                .IsRequired()
            ;

        }
    }
}
