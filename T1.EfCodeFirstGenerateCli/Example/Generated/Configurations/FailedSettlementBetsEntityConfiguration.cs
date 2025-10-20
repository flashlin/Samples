using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class FailedSettlementBetsEntityConfiguration : IEntityTypeConfiguration<FailedSettlementBetsEntity>
    {
        public void Configure(EntityTypeBuilder<FailedSettlementBetsEntity> builder)
        {
            builder.ToTable("FailedSettlementBets");


            builder.Property(x => x.Id)
                .HasColumnType("bigint(19,0)")
                .IsRequired()
            ;

            builder.Property(x => x.TransId)
                .HasColumnType("bigint(19,0)")
                .IsRequired()
            ;

            builder.Property(x => x.MatchResultId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.Action)
                .HasColumnType("varchar(100)")
                .IsRequired()
                .HasMaxLength(100)
            ;

            builder.Property(x => x.CreatedOn)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.Reason)
                .HasColumnType("nvarchar")
            ;

            builder.Property(x => x.CreatedBy)
                .HasColumnType("nvarchar(200)")
                .HasMaxLength(200)
            ;

        }
    }
}
