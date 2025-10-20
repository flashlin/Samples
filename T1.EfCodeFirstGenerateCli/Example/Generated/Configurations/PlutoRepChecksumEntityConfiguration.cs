using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class PlutoRepChecksumEntityConfiguration : IEntityTypeConfiguration<PlutoRepChecksumEntity>
    {
        public void Configure(EntityTypeBuilder<PlutoRepChecksumEntity> builder)
        {
            builder.ToTable("PlutoRepChecksum");

            builder.HasKey(x => x.TableName);

            builder.Property(x => x.Id)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.TableName)
                .HasColumnType("varchar(50)")
                .ValueGeneratedOnAdd()
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.LastSyncTimeStamp)
                .HasColumnType("binary(8)")
                .HasMaxLength(8)
            ;

            builder.Property(x => x.LastSyncDateTime)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.LastSyncId)
                .HasColumnType("bigint(19,0)")
            ;

            builder.Property(x => x.LastSyncCount)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.LastRepExeTime)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.LastChecksumTimeStampMin)
                .HasColumnType("binary(8)")
                .HasMaxLength(8)
            ;

            builder.Property(x => x.LastChecksumDateTimeMin)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.LastChecksumIdMin)
                .HasColumnType("bigint(19,0)")
            ;

            builder.Property(x => x.LastChecksumTimeStampMax)
                .HasColumnType("binary(8)")
                .HasMaxLength(8)
            ;

            builder.Property(x => x.LastChecksumDateTimeMax)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.LastChecksumIdMax)
                .HasColumnType("bigint(19,0)")
            ;

            builder.Property(x => x.LastChecksumCountMain)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.LastChecksumValueMain)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.LastChecksumCountRep)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.LastChecksumValueRep)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.LastChecksumExeTime)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.NextChecksumTimeStamp)
                .HasColumnType("binary(8)")
                .HasMaxLength(8)
            ;

            builder.Property(x => x.NextChecksumDatetime)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.NextChecksumId)
                .HasColumnType("bigint(19,0)")
            ;

            builder.Property(x => x.LastAdminCheckSumTS)
                .HasColumnType("binary(8)")
                .HasMaxLength(8)
            ;

            builder.Property(x => x.LastAdminCheckSumTime)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.LastAdminCheckSumId)
                .HasColumnType("bigint(19,0)")
            ;

        }
    }
}
