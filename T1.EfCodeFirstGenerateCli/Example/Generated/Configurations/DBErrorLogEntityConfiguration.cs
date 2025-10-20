using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class DBErrorLogEntityConfiguration : IEntityTypeConfiguration<DBErrorLogEntity>
    {
        public void Configure(EntityTypeBuilder<DBErrorLogEntity> builder)
        {
            builder.ToTable("DBErrorLog");


            builder.Property(x => x.LogId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.KeyName)
                .HasColumnType("varchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.KeyId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.ErrorCode)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.CreatedBy)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.CreatedOn)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.Message)
                .HasColumnType("varchar(100)")
                .IsRequired()
                .HasMaxLength(100)
            ;

        }
    }
}
