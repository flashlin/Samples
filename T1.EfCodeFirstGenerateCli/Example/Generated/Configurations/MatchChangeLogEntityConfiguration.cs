using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class MatchChangeLogEntityConfiguration : IEntityTypeConfiguration<MatchChangeLogEntity>
    {
        public void Configure(EntityTypeBuilder<MatchChangeLogEntity> builder)
        {
            builder.ToTable("MatchChangeLog");


            builder.Property(x => x.MatchResultId)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.EventStatus)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.CreatedOn)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.TStamp)
                .HasColumnType("timestamp")
                .IsRequired()
            ;

        }
    }
}
