using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class MoveHistoryDBJobConfigEntityConfiguration : IEntityTypeConfiguration<MoveHistoryDBJobConfigEntity>
    {
        public void Configure(EntityTypeBuilder<MoveHistoryDBJobConfigEntity> builder)
        {
            builder.ToTable("MoveHistoryDBJobConfig");


            builder.Property(x => x.ActionType)
                .HasColumnType("varchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.LastActionDate)
                .HasColumnType("smalldatetime")
                .IsRequired()
            ;

            builder.Property(x => x.ModifiedOn)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.ModifiedBy)
                .HasColumnType("varchar(100)")
                .HasMaxLength(100)
            ;

        }
    }
}
