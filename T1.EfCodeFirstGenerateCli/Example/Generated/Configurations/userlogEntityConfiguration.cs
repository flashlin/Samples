using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class userlogEntityConfiguration : IEntityTypeConfiguration<userlogEntity>
    {
        public void Configure(EntityTypeBuilder<userlogEntity> builder)
        {
            builder.ToTable("userlog");

            builder.HasKey(x => x.logid);

            builder.Property(x => x.logid)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.userid)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.doing)
                .HasColumnType("nvarchar(300)")
                .HasMaxLength(300)
            ;

            builder.Property(x => x.logdate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.tstamp)
                .HasColumnType("timestamp")
            ;

        }
    }
}
