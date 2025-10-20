using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class PlutoRepChecksumLogEntityConfiguration : IEntityTypeConfiguration<PlutoRepChecksumLogEntity>
    {
        public void Configure(EntityTypeBuilder<PlutoRepChecksumLogEntity> builder)
        {
            builder.ToTable("PlutoRepChecksumLog");


            builder.Property(x => x.TableName)
                .HasColumnType("varchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.Count)
                .HasColumnType("int(10,0)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.Remark)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.SourceCount)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.SourceSum)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.RepCount)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.RepSum)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.TStampMin)
                .HasColumnType("binary(8)")
                .HasMaxLength(8)
            ;

            builder.Property(x => x.TStampMax)
                .HasColumnType("binary(8)")
                .HasMaxLength(8)
            ;

            builder.Property(x => x.MTimeMin)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.MTimeMax)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.CreatedOn)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.CreatedBy)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

        }
    }
}
