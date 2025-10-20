using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class BatchResultHistoryEntityConfiguration : IEntityTypeConfiguration<BatchResultHistoryEntity>
    {
        public void Configure(EntityTypeBuilder<BatchResultHistoryEntity> builder)
        {
            builder.ToTable("BatchResultHistory");


            builder.Property(x => x.ID)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.BatchId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.TotalBets)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.TotalSuccessBets)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.StatusUpdated)
                .HasColumnType("bit")
            ;

            builder.Property(x => x.Action)
                .HasColumnType("varchar(20)")
                .HasMaxLength(20)
            ;

            builder.Property(x => x.IsSuccess)
                .HasColumnType("bit")
            ;

            builder.Property(x => x.StartTime)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.EndTime)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.CreatedOn)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.Remark)
                .HasColumnType("varchar(100)")
                .HasMaxLength(100)
            ;

            builder.Property(x => x.KafkaInfo)
                .HasColumnType("varchar(150)")
                .HasMaxLength(150)
            ;

        }
    }
}
